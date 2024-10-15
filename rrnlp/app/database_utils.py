from collections import Counter
from os.path import abspath, dirname, join
from typing import List, Tuple

import itertools
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


def load_config():
    config_file = join(dirname(abspath(__file__)), './demo_pw.yml')
    print(f'Loading config from {config_file}')
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)
    st.session_state.config = config
    st.session_state['db_path'] = config['db_path']
    # TODO merge with the above db?
    st.session_state['pubmed_db_path'] = config['pubmed_db_path']
    # TODO verify with each of these that the user can write to this topic...
    st.session_state['loaded_config'] = True

def get_db(fancy_row_factory=True, db_path=None):

    def namedtuple_factory(cursor, row):
        return dict(zip([x[0] for x in cursor.description], row))

    if db_path is None:
        db_path = st.session_state.db_path

    cur = sqlite3.connect(
        f'file:{db_path}?mode=rw',
        uri=True,
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    if fancy_row_factory:
        cur.row_factory = namedtuple_factory
    return cur


# TODO should this be cached?
def close_db(db, e=None):
    #if app.db is not None:
    #    app.db.close()
    #cur = g.pop('db', None)

    if db is not None:
        db.close()

def current_topics(uid):
    cur = get_db(True)
    res = cur.execute('''select topic_uid, topic_name, search_text, search_query, final from user_topics where uid=?''', (uid,))
    all_topics = res.fetchall()
    close_db(cur)
    return all_topics


# TODO consider distinguishing between the automatically generated query and the actually searched query
def get_next_topic_uid(uid='', topic_name='', search_text='', query='', final=0):
    if uid is None or (isinstance(uid, str) and len(uid) == 0):
        assert len(topic_name) == 0, 'user id must be defined before writing any information'
        assert len(search_text) == 0, 'user id must be defined before writing any information'
        assert len(query) == 0, 'user id must be defined before writing any information'

    topic_name = topic_name.replace('"', "''")
    search_text = search_text.replace('"', "''")
    query = query.replace('"', "''")
    cur = get_db(False)
    res = cur.execute('''select MAX(topic_uid) from user_topics;''')
    res = res.fetchone()[0]
    if res is None:
        next_topic = 0
    else:
        next_topic = res + 1

    # TODO do these need to be escaped?
    if uid:
        cur.execute('''
                INSERT INTO
                user_topics(topic_uid, uid, topic_name, search_text, search_query, final)
                VALUES(?, ?, ?, ?, ?, ?);
            ''', (next_topic, uid, topic_name, search_text, query, final)
        )
        cur.commit()
    close_db(cur)
    return next_topic


# TODO should this include whether the cochrane filter got used?
def write_topic_info(topic_uid, uid, topic_name, search_prompt, query, final):
    cur = get_db(False)
    topic_name = topic_name.replace('"', "''")
    search_prompt = search_prompt.replace('"', "''")
    query = query.replace('"', "''")
    # TODO do these need to be escaped?
    cur.execute('''
        INSERT OR REPLACE INTO user_topics(topic_uid, uid, topic_name, search_text, search_query, final)
        VALUES(?,?,?,?,?, ?);
    ''',(topic_uid, uid, topic_name, search_prompt, query, final))
    cur.commit()
    close_db(cur)

# TODO does this need a user id?
def get_topic_info(topic_uid):
    cur = get_db(True)
    res = cur.execute('''
        select topic_uid, uid, topic_name, search_text, search_query, final from user_topics
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
def perform_pubmed_search(pubmed_query, topic_uid, persist, run_ranker=False):
    # TODO allow reset the search results?
    # first: perform the search and get the result document ids
    try:
        pubmed_query = pubmed_query.replace("''", '"')
        count, pmids = SearchBot.PubmedQueryGeneratorBot.execute_pubmed_search(pubmed_query, retmax=10000)
        pmids = set(pmids)
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
    else:
        # fetch any manual insertions
        existing_pmids_and_screening_results = fetch_pmids_and_screening_results(topic_uid)
        screening_results = pd.DataFrame.from_records(existing_pmids_and_screening_results)
        print('src', screening_results.columns)
        if len(screening_results) > 0:
            new_pmids = pmids - set(screening_results['pmid'])
            search_results_df = pd.DataFrame.from_records([{'pmid': pmid, 'human_decision': 'Unscreened', 'robot_ranking': None} for pmid in new_pmids])
            screening_results = pd.concat([screening_results, search_results_df], axis=1)
        else:
            screening_results = pd.DataFrame.from_records([{'pmid': pmid, 'human_decision': 'Unscreened', 'robot_ranking': None} for pmid in pmids])


    screening_results['pmid'] = screening_results['pmid'].astype('int64')
    #print('sdf', screening_results.columns)
    assert len(set(screening_results['pmid'].tolist()) & set(map(int, pmids))) > 0

    # TODO this aggregation operation should happen in the db utils source function since it ties most of these elements together
    # TODO this should really get pushed all the way into sqlite. it only happens because, well, pubmed has the same article multiple times (so it seems)
    # get any meta-data
    article_data = get_pubmed_article_data(list(set(screening_results['pmid'])))
    article_data_df = pd.DataFrame.from_records(article_data)

    #_setify = lambda xs: '; '.join(list(filter(lambda y: len(y) > 0, itertools.chain.from_iterable((map(str.strip, x.split(';')) for x in xs)))))

    article_data_df = article_data_df.groupby('pmid', as_index=False).agg(lambda x: x.iloc[0]).reset_index()

    #article_data_df = article_data_df.groupby('pmid', as_index=False).agg({
    #    'pmid': 'max',
    #    'title': 'max',
    #    'abstract': 'max',
    #    'pubdate': _take_longest,
    #    'mesh_terms': _setify,
    #    'publication_types': _setify,
    #    'journal': _setify,
    #    # no space after the semicolon. Also not an ideal method for joining, but close enough.
    #    'authors': _setify,
    #    'chemical_list': _setify,
    #    'keywords': _setify,
    #    'doi': lambda x: '; '.join(x),
    #}).reset_index()
    article_data_df['pmid'] = article_data_df['pmid'].astype('int64')
    article_data_pmids = set(article_data_df['pmid'].apply(int).tolist())
    search_pmids = set(map(int, pmids))
    assert len(article_data_pmids & search_pmids) > 0
    assert len(article_data) > 0, f"Error in retrieving article data"

    df = pd.merge(screening_results, article_data_df, on='pmid', how='outer')
    if run_ranker:
        ranking_start = time.time()
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
        topic = st.session_state.topic_information['search_prompt']
        print('topic', topic)
        pmid_scores = screener.predict_for_topic(topic, list(map(str, df['pmid'].to_list())))
        pmid_scores = [(int(x[0]), x[1]) for x in pmid_scores]
        print('scores sample', pmid_scores[:10])
        pmid_scores = dict(pmid_scores)
        df['robot_ranking'] = [pmid_scores.get(pmid, None) for pmid in df['pmid']]
        article_data_df['robot_ranking'] = [pmid_scores.get(pmid, None) for pmid in article_data_df['pmid']]
        ranking_end = time.time()
        print(f'AutoRanking took {ranking_end - ranking_start} seconds')
        df = df.sort_values(by='robot_ranking', ascending=False, na_position='last')
        print('df samples', df[:10].to_dict(orient='records'))
        print('article_data_df samples', article_data_df[:10].to_dict(orient='records'))
    # TODO sort by screener rating
    return count, pmids, article_data_df, df, None


def run_robot_ranker(topic_uid):
    # TODO active learning storage?
    start_time = time.time()
    cur = get_db(False)
    res = cur.execute('''select search_text from user_topics where topic_uid=?''', (topic_uid,))
    topic = res.fetchone()[0]

    res = cur.execute('''
        SELECT pmid FROM search_screening_results
        WHERE search_screening_results.topic_uid LIKE ?
        ORDER BY search_screening_results.robot_ranking DESC
    ''', (topic_uid,))
    pmids = res.fetchall()
    pmids = [x[0] for x in pmids]
    #pmids = [x['pmid'] for x in pmids]
    index_choice = st.session_state.config['default_pubmed_index']
    default_weights = st.session_state.config['pubmed_indexes'][index_choice]['base_weights']
    index_path = st.session_state.config['pubmed_indexes'][index_choice]['embeddings_path']
    pmids_path = st.session_state.config['pubmed_indexes'][index_choice]['pmids_path']

    # TODO should this be cached?
    screener = ScreenerBot.load_screener(
        weights=default_weights,
        embeddings_path=index_path,
        pmids_positions=pmids_path,
        device='cpu',
    )
    pmid_scores = screener.predict_for_topic(topic, pmids)
    print('scores sample', pmid_scores[:20])
    #pmid_score_positions = screener.predict_for_topic(topic, pmids)
    #ranked_pmids, scores, positions = zip(*pmid_score_positions)
    #ranked_pmids, scores = zip(*(x['pmid'], x['distance'] for x in pmid_score_positions))
    ranked_pmids, scores = zip(*pmid_scores)

    update_list = list(zip(scores, itertools.cycle([topic_uid]), ranked_pmids))
    cur.executemany('''
        UPDATE search_screening_results SET robot_ranking=? WHERE topic_uid=? and pmid LIKE ?
    ''', update_list)
    cur.commit()
    print(update_list[:10])
    end_time = time.time()
    print(f'Updating {len(pmid_scores)} autoranking took {end_time - start_time} seconds')


def get_topic_pubmed_results(topic_uid):
    cur = get_db(True)
    # TODO do these need to be escaped?
    res = cur.execute('''
        SELECT user_topics(topic_uid, pmid, human_decision, robot_ranking)
        WHERE topic_uid LIKE (?,)
        ORDER BY robot_ranking DESC
    ''', (topic_uid,))
    results = res.fetchall()
    close_db(cur)
    return results


def fetch_pmids_and_screening_results(topic_uid):
    cur = get_db(True)
    res = cur.execute('''
        SELECT pmid, human_decision, robot_ranking
        FROM search_screening_results
        WHERE topic_uid = ?
        ORDER BY robot_ranking DESC
    ''', (topic_uid,))
    results = res.fetchall()
    close_db(cur)
    return results

def insert_unscreened_pmids(topic_uid, pmids):
    cur = get_db(True)
    cur.executemany('''
        INSERT OR REPLACE INTO search_screening_results(topic_uid, pmid) VALUES(?, ?)
    ''', [(topic_uid, str(pmid)) for pmid in pmids])
    cur.commit()
    res = cur.execute('''
        SELECT pmid, human_decision, robot_ranking
        FROM search_screening_results
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
    #article_data_df['pmid'] = article_data_df['pmid'].astype('int64')
    #article_data_pmids = set(article_data_df['pmid'].apply(int).tolist())
    #search_pmids = set(map(int, pmids))

    screening_results['pmid'] = screening_results['pmid'].astype(str)
    article_data_df['pmid'] = article_data_df['pmid'].astype(str)
    df = pd.merge(screening_results, article_data_df, on='pmid', how='outer')
    df = df.sort_values(by='robot_ranking', ascending=False, na_position='last')
    # TODO sort by screener rating
    return len(pmids), pmids, article_data_df, df

def insert_topic_human_screening_pubmed_results(topic_uid, pmid_to_human_screening: dict, source='automatic'):
    cur = get_db(False)
    # TODO do these need to be escaped?
    decisions = Counter(pmid_to_human_screening.values())
    print(f'attempting to insert {len(pmid_to_human_screening)}; {decisions} screening decisions for topic {topic_uid}')
    cur.executemany('''
        INSERT OR REPLACE INTO search_screening_results(topic_uid, pmid, human_decision, source)
        VALUES(:ti, :pm, :hs, :src)
    ''', [{'hs': human, 'ti': topic_uid, 'pm': pmid, 'src': source} for (pmid, human) in pmid_to_human_screening.items()])
    cur.commit()
    close_db(cur)

def get_pubmed_article_data(pmids: list):
    cur = get_db(True, db_path=st.session_state['pubmed_db_path'])
        #SELECT pmid,title,abstract,pubdate,mesh_terms,publication_types,issue,pages,journal,authors,chemical_list,keywords,doi,pmc,other_id,medline_ta,nlm_unique_id,issn_linking,country,grant_ids
    print(f'loading database {st.session_state["pubmed_db_path"]} to get records for {len(pmids)} pmids')
    #query = f'''
    #    SELECT DISTINCT pmid,title,abstract,pubdate,mesh_terms,publication_types,journal,authors,chemical_list,keywords,doi
    #    FROM pubmed_data
    #    WHERE pmid IN ({','.join(map(str, pmids))})
    #'''
    query = f'''
        SELECT DISTINCT pmid,titles,abstracts
        FROM pubmed_data
        WHERE pmid IN ({','.join(map(str, pmids))})
    '''
    #print(query)
    res = cur.execute(query)
    res = res.fetchall()
    close_db(cur)
    return res

def get_auto_evidence_map_from_topic_uid(topic_uid):
    # TODO should this be joined with screening?
    cur = get_db(True)
    query = '''
        SELECT pubmed_extractions_ico_re.pmid, pubmed_extractions_ico_re.intervention from pubmed_extractions_ico_re
        INNER JOIN search_screening_results ON pubmed_extractions_ico_re.pmid = search_screening_results.pmid
        WHERE topic_uid = :topic_uid
    '''
        #AND search_screening_results.human_decision = "Include"
    print(query, topic_uid)
    pmids = cur.execute(query, {'topic_uid': topic_uid})
    pmids = pmids.fetchall()
    print(f'found {len(pmids)} pmids')
            # TODO add this to the database
            #pubmed_extractions_ico_re.label,
            #pubmed_extractions_ico_re.population,
            #pubmed_extractions_ico_re.sample_size,
            #pubmed_extractions.prob_low_rob,
            #pubmed_extractions.low_rsg_bias,
            #pubmed_extractions.low_ac_bias,
            #pubmed_extractions.low_bpp_bias
    query = '''
        SELECT
            pubmed_extractions_ico_re.pmid,
            pubmed_extractions.title AS title,
            pubmed_extractions.abstract as abstract,
            pubmed_extractions_ico_re.intervention,
            pubmed_extractions_ico_re.comparator,
            pubmed_extractions_ico_re.outcome,
            pubmed_extractions_ico_re.evidence,
            pubmed_extractions.p,
            pubmed_extractions.num_randomized,
            pubmed_extractions.prob_lob_rob
        FROM pubmed_extractions_ico_re
        INNER JOIN search_screening_results ON pubmed_extractions_ico_re.pmid = search_screening_results.pmid
        INNER JOIN pubmed_extractions ON pubmed_extractions.pmid = search_screening_results.pmid
        WHERE search_screening_results.topic_uid = :topic_uid
        AND search_screening_results.human_decision = "Include"
        ORDER BY search_screening_results.robot_ranking DESC
    '''
    print(query)
    res = cur.execute(query, {'topic_uid': topic_uid})
    selections = res.fetchall()
    close_db(cur)
    return pmids, selections

def insert_numerical_extractions(extraction: List[Tuple]):
    cur = get_db(True)
    query = '''
        INSERT or REPLACE INTO pubmed_extractions_numerical(pmid, intervention, comparator, outcome, outcome_type, binary_result, continuous_result) VALUES(?,?,?,?,?,?,?)
        '''
    cur.executemany(query, extractions)
    cur.commit()

def get_numerical_extractions_for_topic(topic_uid):
    cur = get_db(True)
    query = '''
        SELECT
            pubmed_extractions_ico_re.pmid,
            pubmed_extractions.title AS title,
            pubmed_extractions.abstract as abstract,
            pubmed_extractions_numerical.intervention,
            pubmed_extractions_numerical.comparator,
            pubmed_extractions_numerical.outcome,
            pubmed_extractions_numerical.outcome_type,
            pubmed_extractions_numerical.binary_result,
            pubmed_extractions_numerical.continuous_result
        FROM pubmed_extractions_numerical
        INNER JOIN search_screening_results ON pubmed_extractions_numerical.pmid = search_screening_results.pmid
        INNER JOIN pubmed_extractions ON pubmed_extractions.pmid = search_screening_results.pmid
        WHERE search_screening_results.topic_uid = :topic_uid
        AND search_screening_results.human_decision = "Include"
        ORDER BY search_screening_results.robot_ranking DESC
    '''
    res = cur.execute(query, {'topic_uid': topic_uid})
    selections = res.fetchall()
    close_db(cur)
    return selections

def get_auto_evidence_map_from_pmids(pmids):
    # TODO should this be joined with screening?
    cur = get_db(True)
    res = cur.executemany('''
        SELECT pmid, title, abstract, intervention, comparator, outcome, label, evidence, population, sample_size, prob_low_rob, low_rsg_bias,low_ac_bias,low_bpp_bias
        FROM pubmed_extractions_ico_re
        INNER_JOIN search_screening_results USING pmid
        WHERE pmid = ?
        AND human_decision = 'Include'
        ORDER BY search_screening_results.robot_ranking DESC
    ''', pmids)
    selections = res.fetchall()
    close_db(cur)
    return selections

if not st.session_state.get('loaded_config', False):
    load_config()
