from collections import Counter
from os.path import abspath, dirname, join
from typing import List, Tuple

import datetime
import itertools
import json
import os
import random
import sqlite3
import time
import urllib

import pandas as pd
import streamlit as st
import yaml

from yaml.loader import SafeLoader

# todo the functions used here should probably migrate to this file?
import rrnlp.models.SearchBot as SearchBot
import rrnlp.models.ScreenerBot as ScreenerBot

PRAGMA_WAL='PRAGMA journal_mode=wal;'
PRAGMA_DEL='PRAGMA journal_model=delete;'


def load_config():
    config_file = join(dirname(abspath(__file__)), './demo_pw.yml')
    print(f'Loading config from {config_file}')
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)
    st.session_state.config = config
    st.session_state['source_db_path'] = config['source_db_path']
    st.session_state['user_data_db_path'] = config['user_data_db_path']
    # TODO merge with the above db?
    st.session_state['pubmed_data_db_path'] = config['pubmed_data_db_path']
    st.session_state['openai_api_key'] = config['openai_api_key']
    # TODO verify with each of these that the user can write to this topic...
    st.session_state['loaded_config'] = True

@st.cache_resource
def load_default_screener():
    # TODO should this be cached?
    index_choice = st.session_state.config['default_pubmed_index']
    default_weights = st.session_state.config['pubmed_indexes'][index_choice]['base_weights']
    index_path = st.session_state.config['pubmed_indexes'][index_choice]['embeddings_path']
    pmids_path = st.session_state.config['pubmed_indexes'][index_choice]['pmids_path']
    screener = ScreenerBot.load_screener(
        weights=default_weights,
        embeddings_path=index_path,
        pmids_positions=pmids_path,
        device='cpu',
    )
    return screener

def _custom_screener_location(topic_uid, create=False):
    path = os.path.join(st.session_state.config['fine_tuned_screener_home'], f'topic_{topic_uid}')
    return path

def load_screener(topic_uid, force_default=True):
    screener_home = _custom_screener_location(topic_uid)
    if force_default or not os.path.exists(screener_home):
        return load_default_screener()
    else:
        index_choice = st.session_state.config['default_pubmed_index']
        index_path = st.session_state.config['pubmed_indexes'][index_choice]['embeddings_path']
        pmids_path = st.session_state.config['pubmed_indexes'][index_choice]['pmids_path']
        screener = ScreenerBot.load_screener(
            weights=_custom_screener_location(),
            embeddings_path=index_path,
            pmids_positions=pmids_path,
            device='cpu',
        )
        return screener

def get_db(fancy_row_factory=True, source_db_path=None, user_data_db=None, pubmed_data_db=None, pragma_cmd=None):

    def namedtuple_factory(cursor, row):
        return dict(zip([x[0] for x in cursor.description], row))

    if source_db_path is None:
        source_db_path = st.session_state.source_db_path
    if user_data_db is None:
        user_data_db = st.session_state.user_data_db_path
    if pubmed_data_db is None:
        pubmed_data_db = st.session_state.pubmed_data_db_path
    print(f'connecting to {source_db_path}, {user_data_db}, {pubmed_data_db}')

    cur = sqlite3.connect(
        f'file:{source_db_path}?mode=rw',
        uri=True,
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    cur.execute(f'ATTACH "{user_data_db}" as user_db')
    cur.execute(f'ATTACH "{pubmed_data_db}" as pubmed_db')
    if fancy_row_factory:
        cur.row_factory = namedtuple_factory
    if pragma_cmd is not None:
        cur.execute(pragma_cmd)
    return cur


def close_db(db, e=None):
    if db is not None:
        db.close()

def current_topics(uid):
    cur = get_db(True, pragma_cmd=PRAGMA_DEL)
    res = cur.execute('''select topic_uid, topic_name, search_text, search_query, generated_query, final from user_db.user_topics where uid=?''', (uid,))
    all_topics = res.fetchall()
    close_db(cur)
    return all_topics

def get_next_topic_uid(
        uid='',
        topic_name='',
        search_text='',
        search_query='',
        generated_query='',
        used_cochrane_filter=0,
        used_robot_reviewer_rct_filter=0,
        final=0,
    ):
    if uid is None or (isinstance(uid, str) and len(uid) == 0):
        assert len(topic_name) == 0, 'user id must be defined before writing any information'
        assert len(search_text) == 0, 'user id must be defined before writing any information'
        assert len(query) == 0, 'user id must be defined before writing any information'

    topic_name = topic_name.replace('"', "''")
    search_text = search_text.replace('"', "''")
    query = search_query.replace('"', "''")
    cur = get_db(False, pragma_cmd=PRAGMA_DEL)
    res = cur.execute('''select MAX(topic_uid) from user_db.user_topics;''')
    res = res.fetchone()[0]
    if res is None:
        next_topic = 0
    else:
        next_topic = res + 1

    # TODO do these need to be escaped?
    if uid:
        cur.execute('''
                INSERT INTO
                user_db.user_topics(topic_uid, uid, topic_name, search_text, search_query, generated_query, used_cochrane_filter, used_robot_reviewer_rct_filter, final)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);
            ''', (next_topic, uid, topic_name, search_text, search_query, generated_query, used_cochrane_filter, used_robot_reviewer_rct_filter, final)
        )
        cur.commit()
    close_db(cur)
    return next_topic


def write_topic_info(
        topic_uid,
        uid,
        topic_name,
        search_text,
        search_query,
        generated_query,
        used_cochrane_filter,
        used_robot_reviewer_rct_filter,
        final,
    ):
    topic_name = topic_name.replace('"', "''")
    pt = search_text.replace('"', "''")
    search_query = search_query.replace('"', "''")
    generated_query = generated_query.replace('"', "''")
    if topic_uid is None:
        topic_uid = get_next_topic_uid(
            uid=uid,
            topic_name=topic_name,
            search_text=search_text,
            search_query=search_query,
            generated_query=generated_query,
            used_cochrane_filter=used_cochrane_filter,
            used_robot_reviewer_rct_filter=used_robot_reviewer_rct_filter,
            final=final,
        )
    cur = get_db(False, pragma_cmd=PRAGMA_DEL)
    # TODO do these need to be escaped?
    cur.execute('''
        INSERT OR REPLACE INTO
        user_db.user_topics(topic_uid, uid, topic_name, search_text, search_query, generated_query, used_cochrane_filter, used_robot_reviewer_rct_filter, final)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);
        ''', (topic_uid, uid, topic_name, search_text, search_query, generated_query, used_cochrane_filter, used_robot_reviewer_rct_filter, final)
    )
    cur.commit()
    close_db(cur)
    return topic_uid

# TODO does this need a user id?
def get_topic_info(topic_uid):
    cur = get_db(True, pragma_cmd=PRAGMA_WAL)
    res = cur.execute('''
        select topic_uid, uid, topic_name, search_text, search_query, generated_query, used_cochrane_filter, used_robot_reviewer_rct_filter, final from user_db.user_topics
        where topic_uid=?
    ''',(topic_uid,))
    results = res.fetchall()
    assert len(results) == 1
    close_db(cur)
    return results[0]

def _take_longest(strs):
    strs = sorted(strs, key=lambda x: len(x), reverse=True)
    return strs[0]

def _setify(xs):
    parts = []
    for x in xs:
        parts.append(list(map(str.strip, x.split(';'))))
    parts = list(filter(lambda x: len(x) > 0, itertools.chain.from_iterable(parts)))
    return '; '.join(set(parts))

@st.cache_data
def perform_pubmed_search(pubmed_query, topic_uid, persist, run_ranker=False, fetch_all_by_date=False):
    # TODO allow reset the search results?
    # first: perform the search and get the result document ids
    try:
        pubmed_query = pubmed_query.replace("''", '"')
        count, pmids = SearchBot.PubmedQueryGeneratorBot.execute_pubmed_search(pubmed_query, retmax=10000, fetch_all_by_date=fetch_all_by_date)
        pmids = set(map(str, pmids))
        count = int(count)
        print(f'retrieved {len(pmids)} pmids of {count}')
    except urllib.error.HTTPError as e:
        print(e)
        return None, None, None, None, e
    if count == 0:
        # TODO should really pass this up
        #st.markdown('Found no results for this search, generate a new search before committing to this one!')
        return 0, [], None, None, None
    # get any other ids and add these to the screening, return the results
    if persist:
        existing_pmids_and_screening_results = insert_unscreened_pmids(topic_uid, pmids)
        screening_results = pd.DataFrame.from_records(existing_pmids_and_screening_results)
        screening_results['pmid'] = screening_results['pmid'].astype(str)
    else:
        # fetch any manual insertions
        existing_pmids_and_screening_results = fetch_pmids_and_screening_results(topic_uid)
        screening_results = pd.DataFrame.from_records(existing_pmids_and_screening_results)
        if len(screening_results) > 0:
            screening_results['pmid'] = screening_results['pmid'].astype(str)
            new_pmids = pmids - set(screening_results['pmid'])
            if len(new_pmids) > 0:
                search_results_df = pd.DataFrame.from_records([{'pmid': str(pmid), 'human_decision': 'Unscreened', 'robot_ranking': None} for pmid in new_pmids])
                og_columns = screening_results.columns
                assert set(screening_results.columns) == set(search_results_df.columns), str(screening_results.columns) + " " + str(search_results_df.columns)
                screening_results = screening_results[screening_results['pmid'].isin(new_pmids)]
                screening_results = pd.concat([screening_results, search_results_df], axis=0)
        else:
            screening_results = pd.DataFrame.from_records([{'pmid': pmid, 'human_decision': 'Unscreened', 'robot_ranking': None} for pmid in pmids])
            new_pmids = pmids


    pmids = screening_results['pmid'].tolist()
    print(f'Have {len(pmids)} pmids, total {type(pmids)}\n{pmids[:10]}')

    # TODO this aggregation operation should happen in the db utils source function since it ties most of these elements together
    # TODO this should really get pushed all the way into sqlite. it only happens because, well, pubmed has the same article multiple times (so it seems)
    # get any meta-data
    article_data = get_pubmed_article_data(list(set(screening_results['pmid'])))
    article_data_df = pd.DataFrame.from_records(article_data)

    #_setify = lambda xs: '; '.join(list(filter(lambda y: len(y) > 0, itertools.chain.from_iterable((map(str.strip, x.split(';')) for x in xs)))))

    article_data_df = article_data_df.groupby('pmid', as_index=False).agg(lambda x: x.iloc[0]).reset_index()

    article_data_df['pmid'] = article_data_df['pmid'].astype(str)
    article_data_pmids = set(article_data_df['pmid'].apply(int).tolist())
    search_pmids = set(map(int, pmids))
    assert len(article_data_pmids & search_pmids) > 0
    assert len(article_data) > 0, f"Error in retrieving article data"

    df = pd.merge(screening_results, article_data_df, on='pmid', how='outer')
    if run_ranker or run_ranker == 1:
        ranking_start = time.time()
        topic = st.session_state.topic_information['search_text']
        screener = load_screener(topic_uid)
        print('topic', topic)
        pmid_scores = screener.predict_for_topic(topic, list(map(str, df['pmid'].to_list())))
        pmid_scores = [(str(x[0]), x[1]) for x in pmid_scores]
        pmid_scores = dict(pmid_scores)
        df['robot_ranking'] = [pmid_scores.get(str(pmid), None) for pmid in df['pmid']]
        article_data_df['robot_ranking'] = [pmid_scores.get(pmid, None) for pmid in article_data_df['pmid']]
        ranking_end = time.time()
        print(f'AutoRanking took {ranking_end - ranking_start} seconds')
        df = df.sort_values(by='robot_ranking', ascending=False, na_position='last')
        if persist:
            update_list = list(zip(df['robot_ranking'], itertools.cycle([topic_uid]), df['pmid']))
            cur = get_db(pragma_cmd=PRAGMA_WAL)
            cur.executemany('''
                UPDATE user_db.search_screening_results
                SET robot_ranking=?
                WHERE topic_uid=? AND pmid LIKE ?
            ''', update_list)
            cur.commit()

    # TODO sort by screener rating
    # TODO sort the articles with missing information to the bottom?
    return count, pmids, article_data_df, df, None


def run_robot_ranker(topic_uid):
    # TODO active learning storage?
    start_time = time.time()
    cur = get_db(False, pragma_cmd=PRAGMA_WAL)
    res = cur.execute('''select search_text from user_db.user_topics where topic_uid=?''', (topic_uid,))
    topic = res.fetchone()[0]

    res = cur.execute('''
        SELECT pmid FROM user_db.search_screening_results screening
        WHERE screening.topic_uid LIKE ?
        ORDER BY screening.robot_ranking DESC
    ''', (topic_uid,))
    pmids = res.fetchall()
    pmids = [x[0] for x in pmids]

    screener = load_screener(topic_uid)
    pmid_scores = screener.predict_for_topic(topic, pmids)
    print('scores sample', pmid_scores[:20])
    ranked_pmids, scores = zip(*pmid_scores)

    update_list = list(zip(scores, itertools.cycle([topic_uid]), ranked_pmids))
    cur = get_db(False, pragma_cmd=PRAGMA_DEL)
    cur.executemany('''
        UPDATE user_db.search_screening_results SET robot_ranking=? WHERE topic_uid=? and pmid LIKE ?
    ''', update_list)
    cur.commit()
    print(update_list[:10])
    end_time = time.time()
    print(f'Updating {len(pmid_scores)} autoranking took {end_time - start_time} seconds')


def get_topic_pubmed_results(topic_uid):
    cur = get_db(True, pragma_cmd=PRAMGA_WAL)
    # TODO do these need to be escaped?
    res = cur.execute('''
        SELECT user_db.user_topics(topic_uid, pmid, human_decision, robot_ranking)
        WHERE topic_uid LIKE (?,)
        ORDER BY robot_ranking DESC
    ''', (topic_uid,))
    results = res.fetchall()
    close_db(cur)
    return results


def fetch_pmids_and_screening_results(topic_uid):
    cur = get_db(True, pragma_cmd=PRAGMA_WAL)
    res = cur.execute('''
        SELECT pmid, human_decision, robot_ranking
        FROM user_db.search_screening_results
        WHERE topic_uid = ?
        ORDER BY robot_ranking DESC
    ''', (topic_uid,))
    results = res.fetchall()
    close_db(cur)
    return results

def insert_unscreened_pmids(topic_uid, pmids, ranks=None):
    cur = get_db(True, pragma_cmd=PRAGMA_DEL)
    if ranks is None:
        ranks = itertools.cycle([None])
    pmids = list(filter(lambda x: len(x) > 0 and x.isdigit(), map(str.strip, pmids)))
    if len(pmids) > 0:
        cur.executemany('''
            INSERT OR REPLACE INTO user_db.search_screening_results(topic_uid, pmid, robot_ranking) VALUES(?, ?, ?)
        ''', [(topic_uid, str(pmid), rank) for pmid, rank in zip(pmids, ranks)])
        cur.commit()
    res = cur.execute('''
        SELECT pmid, human_decision, robot_ranking
        FROM user_db.search_screening_results
        WHERE topic_uid = ?
        ORDER BY robot_ranking DESC
    ''', (topic_uid,))
    results = res.fetchall()
    close_db(cur)
    return results

def get_persisted_pubmed_search_and_screening_results(topic_uid):
    # first check if this has been finalized
    topic_information = get_topic_info(topic_uid)
    if topic_information['final'] == 0:
        return None
    # then get the information
    existing_pmids_and_screening_results = fetch_pmids_and_screening_results(topic_uid)
    screening_results = pd.DataFrame.from_records(existing_pmids_and_screening_results)
    pmids = list(set(screening_results['pmid']))
    article_data = get_pubmed_article_data(pmids)
    article_data_df = pd.DataFrame.from_records(article_data)
    article_data_df = article_data_df.groupby('pmid', as_index=False).agg(lambda x: x.iloc[0]).reset_index()

    screening_results['pmid'] = screening_results['pmid'].astype(str)
    article_data_df['pmid'] = article_data_df['pmid'].astype(str)
    df = pd.merge(screening_results, article_data_df, on='pmid', how='outer')
    df = df.sort_values(by='robot_ranking', ascending=False, na_position='last')
    return len(pmids), pmids, article_data_df, df

def insert_topic_human_screening_pubmed_results(topic_uid, pmid_to_human_screening: dict, source='automatic'):
    cur = get_db(False, pragma_cmd=PRAGMA_DEL)
    # TODO do these need to be escaped?
    decisions = Counter(pmid_to_human_screening.values())
    print(f'attempting to insert {len(pmid_to_human_screening)}; {decisions} screening decisions for topic {topic_uid}')
    cur.executemany('''
        INSERT INTO user_db.search_screening_results(topic_uid, pmid, human_decision, source)
        VALUES(:ti, :pm, :hs, :src)
        ON CONFLICT(topic_uid, pmid)
        DO UPDATE SET human_decision=:hs
    ''', [{'hs': human, 'ti': topic_uid, 'pm': pmid, 'src': source} for (pmid, human) in pmid_to_human_screening.items()])
    cur.commit()
    close_db(cur)

def get_pubmed_article_data(pmids: list):
    cur = get_db(True, pragma_cmd=PRAGMA_WAL)
    print(f'fetching records for {len(pmids)} pmids')
    query = f'''
        SELECT DISTINCT pmid,title,abstract
        FROM pubmed_db.article_data
        WHERE pmid IN ({','.join(map(str, pmids))})
    '''
    res = cur.execute(query)
    res = res.fetchall()
    close_db(cur)
    return res

def get_auto_evidence_map_from_topic_uid(topic_uid, included_documents_only):
    cur = get_db(True, pragma_cmd=PRAGMA_WAL)

    query = f'''
        SELECT pmid
        FROM user_db.search_screening_results screening
        WHERE screening.topic_uid = {topic_uid}
        AND screening.human_decision = 'Include'
    '''
    included_pmids = set([x['pmid'] for x in cur.execute(query).fetchall()])
    if included_documents_only:
        clause = f'WHERE screening.human_decision = "Include" AND screening.topic_uid = {topic_uid}'
    else:
        clause = f'WHERE screening.topic_uid = {topic_uid}'
          #screening.robot_ranking,
    query = f'''
        SELECT
          screening.pmid,
          screening.human_decision,
          article_data.title,
          article_data.abstract,
          article_data.journal,
          article_data.pubdate,
          article_data.publication_types,
          article_data.keywords,
          article_data.is_rct,
          article_data.prob_rct,
          study_bot.num_randomized,
          study_bot.study_design,
          study_bot.prob_sr,
          study_bot.is_sr,
          study_bot.prob_cohort,
          study_bot.is_cohort,
          study_bot.prob_consensus,
          study_bot.is_consensus,
          study_bot.prob_ct,
          study_bot.is_ct,
          study_bot.prob_ct_protocol,
          study_bot.is_ct_protocol,
          study_bot.prob_guideline,
          study_bot.is_guideline,
          study_bot.prob_qual,
          study_bot.is_qual,
          study_bot.rct_bot_is_rct_sensitive,
          study_bot.rct_bot_is_rct_balanced,
          study_bot.rct_bot_is_rct_precise
        FROM
            user_db.search_screening_results screening
        INNER JOIN
            pubmed_db.article_data article_data
        ON
            screening.pmid = article_data.pmid
        INNER JOIN
            pubmed_db.study_design_bot study_bot
        ON
            screening.pmid = study_bot.pmid
        {clause}
        GROUP BY screening.pmid, screening.human_decision
    '''

    res = cur.execute(query)
    extractions = res.fetchall()
    extractions = pd.DataFrame.from_records(extractions)
    pmids = '(' + ','.join(set(map(str, extractions['pmid']))) + ')'
    query = f'''
        SELECT
        *
        FROM pubmed_db.ico_ev_bot ico_re
        WHERE pmid in {pmids}
    '''
    res = cur.execute(query)
    ico_re = res.fetchall()
    ico_re = pd.DataFrame.from_records(ico_re)
    query = f'''
    SELECT *
    FROM pubmed_db.pico_bot pico_bot
    WHERE pico_bot.pmid in {pmids}
    '''
    res = cur.execute(query)
    picos = res.fetchall()
    picos = pd.DataFrame.from_records(picos)
    close_db(cur)
    return included_pmids, ico_re, picos, extractions

def get_extractions_for_pmids(pmids):
    cur = get_db(True, pragma_cmd=PRAGMA_WAL)
    pmids = '(' + ','.join(map(str, pmids)) + ')'
    query = f'''
        SELECT
        *
        FROM pubmed_db.ico_ev_bot ico_re
        WHERE pmid in {pmids}
    '''
    res = cur.execute(query)
    ico_re = res.fetchall()
    ico_re = pd.DataFrame.from_records(ico_re)

    query = f'''
        SELECT
          article_data.pmid,
          article_data.journal,
          article_data.pubdate,
          article_data.publication_types,
          article_data.keywords,
          article_data.is_rct,
          article_data.prob_rct,
          study_bot.study_design,
          study_bot.is_rct,
          study_bot.prob_rct,
          study_bot.prob_low_rob,
          study_bot.num_randomized,
          study_design,
          study_bot.prob_sr,
          study_bot.is_sr,
          study_bot.prob_cohort,
          study_bot.is_cohort,
          study_bot.prob_consensus,
          study_bot.is_consensus,
          study_bot.prob_ct,
          study_bot.is_ct,
          study_bot.prob_ct_protocol,
          study_bot.is_ct_protocol,
          study_bot.prob_guideline,
          study_bot.is_guideline,
          study_bot.prob_qual,
          study_bot.is_qual,
          study_bot.is_rct,
          study_bot.rct_bot_is_rct,
          study_bot.rct_bot_is_rct_sensitive,
          study_bot.rct_bot_is_rct_balanced,
          study_bot.rct_bot_is_rct_precise
        FROM 
            pubmed_db.article_data article_data
        LEFT JOIN
            pubmed_db.study_design_bot study_bot
        ON
            article_data.pmid = study_bot.pmid
        WHERE article_data.pmid in {pmids}
    '''
    res = cur.execute(query)
    extractions = res.fetchall()
    extractions = pd.DataFrame.from_records(extractions)
    query = f'''
    SELECT *
    FROM pubmed_db.pico_bot pico_bot
    WHERE pico_bot.pmid in {pmids}
    '''
    res = cur.execute(query)
    picos = res.fetchall()
    picos = pd.DataFrame.from_records(picos)
    close_db(cur)
    return ico_re, picos, extractions

# TODO this should probably get moved into utils somehow?
def finetune_ranker(topic_uid):
    screener = load_screener(topic_uid, force_default=True)
    topic_info = get_topic_info(topic_uid)
    topic_text = topic_info['search_text']
    screening_status = fetch_pmids_and_screening_results(topic_uid)
    status_dict = {
        'Include': set(),
        'Exclude': set(),
        'Unscreened': set()
    }
    for row in screening_status:
        status_dict[row['human_decision']].add(row['pmid'])
    counts = {x:len(y) for x,y in status_dict.items()}
    negatives = status_dict['Exclude']
    if len(negatives) < len(status_dict['Include']):
        random.seed(12345)
        # TODO include other negatives if needed!
        # TODO should this sample from the whole DB?
        negatives.update(random.sample(list(status_dict['Unscreened']), k=len(status_dict['Include']) - len(negatives)))

    output_dir = _custom_screener_location(topic_uid)
    if os.path.exists(output_dir):
        os.rename(output_dir, output_dir + datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(output_dir)
    new_screener, losses = screener.finetune_for_annotations(
            topic=topic_text,
            positive_ids=list(status_dict['Include']),
            negative_ids=list(negatives),
            clone_model=True,
    )
    new_screener.save_pretrained(output_dir)
    with open(os.path.join(output_dir, 'losses.json'), 'w') as of:
        of.write(json.dumps(losses, indent=2))
    return new_screener

if not st.session_state.get('loaded_config', False):
    load_config()
