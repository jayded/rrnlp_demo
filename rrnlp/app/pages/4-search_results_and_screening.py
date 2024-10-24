import itertools

from collections import Counter

import pandas as pd
import streamlit as st
st.set_page_config(page_title='Search Results and Screening', layout='wide')

import rrnlp.models.SearchBot as SearchBot

import rrnlp.app.database_utils as database_utils

if 'uid' not in st.session_state:
    st.switch_page('streamlit_app.py')

if 'topic_information' not in st.session_state or 'topic_uid' not in st.session_state.topic_information or 'topic_name' not in st.session_state.topic_information:
    st.switch_page('pages/2-existing_projects.py')

# mystery: why does the st module not...seem cooperate?
if not st.session_state.get('loaded_config', False):
    database_utils.load_config()

cochrane_filter = SearchBot.PubmedQueryGeneratorBot.rct_filter()

# TODO don't use the copies when they can be avoided
search_text = st.session_state.topic_information.get('search_text', '')
search_query = st.session_state.topic_information.get('search_query', '')

# part one: fetch a set of pmids
# if we are still in development
if st.session_state.topic_information['final'] != 1:
    with st.form("search"):
        search_query = st.text_area('Boolean query', value=st.session_state.topic_information.get('search_query', ''))
        submitted_search = st.form_submit_button("Search")
        if submitted_search:
            st.session_state.last_searched = search_query

    if len(search_text) > 0 and st.checkbox(f'Add Cochrane RCT filter? {cochrane_filter}', value=st.session_state.topic_information.get('used_cochrane_filter', False)):
        added_filter = ' AND ' + cochrane_filter
        use_rct_filter = True
    else:
        added_filter = ''
        use_rct_filter = False
    st.session_state.topic_information['use_rct_filter'] = use_rct_filter

    run_ranker = st.checkbox('Run AutoRanker (~1 minute / 5k)?', value=st.session_state.topic_information.get('run_ranker', False))

    st.write('Click "Finalize" to finalize this search and begin screening. Once the search is finalized, no new searches may be added')
    if st.button('Finalize'):
        st.session_state.topic_information['final'] = 1
        finalize = 1
    else:
        finalize = 0

    if (submitted_search or st.session_state.topic_information.get('execute_search', False)) \
        and (search_query != st.session_state.topic_information.get('last_searched', '') or 'df' not in st.session_state.topic_information):
        count, pmids, article_data_df, df, e = database_utils.perform_pubmed_search(search_query + added_filter, st.session_state.topic_information['topic_uid'], persist=st.session_state.topic_information['final']==1, run_ranker=run_ranker, fetch_all_by_date=True)
        st.session_state.topic_information['count'] = count
        st.session_state.topic_information['pmids'] = pmids
        st.session_state.topic_information['article_data_df'] = article_data_df
        st.session_state.topic_information['df'] = df
        st.session_state.topic_information['screening_results'] = df
        st.session_state.topic_information['last_searched'] = search_query
        st.session_state.topic_information['search_query'] = search_query
        st.session_state.topic_information['execute_search'] = False
    else:
        count = st.session_state.topic_information['count']
        pmids = st.session_state.topic_information['pmids']
        article_data_df = st.session_state.topic_information['article_data_df']
        df = st.session_state.topic_information['df']
        e = None
    # we separate these two so the search will be persisted and *then* the topic information updated; doing it the other way means any issues 
    if finalize == 1:
        st.markdown('All pmids from this search will be inserted and persisted into the database')
        database_utils.write_topic_info(
            topic_uid=st.session_state.topic_information['topic_uid'],
            uid=st.session_state.uid,
            topic_name=st.session_state.topic_information['topic_name'],
            search_text=st.session_state.topic_information['search_text'],
            search_query=st.session_state.topic_information['search_query'],
            generated_query=st.session_state.topic_information['generated_query'],
            used_cochrane_filter=st.session_state.topic_information['used_cochrane_filter'],
            used_robot_reviewer_rct_filter=0,
            final=st.session_state.topic_information['final'])
else:
    # if we have finished development and are revisiting this page.
    search_query = st.session_state.topic_information['search_query']
    st.markdown(f'Results for: {search_query}')
    st.write('No more search results can be added via pubmed searches. Add any others manually:')
    with st.form('Insert bulk screening results'):
        st.markdown('Insert a list of pmids to Include. Use spaces or commas to separate them')
        include_pmids = st.text_area('Include pmids', value='', height=20)
        st.markdown('Insert a list of pmids to Exclude. Use spaces or commas to separate them')
        exclude_pmids = st.text_area('Exclude pmids', value='', height=20)
        submitted = st.form_submit_button("Insert")
        if submitted:
            include_pmids = list(itertools.chain.from_iterable([x.split(',') for x in include_pmids.split()]))
            exclude_pmids = list(itertools.chain.from_iterable([x.split(',') for x in exclude_pmids.split()]))
            pmid_to_screening_result = {pmid: 'Include' for pmid in include_pmids}
            pmid_to_screening_result.update({pmid: 'Exclude' for pmid in exclude_pmids})
            database_utils.insert_topic_human_screening_pubmed_results(
                topic_uid=st.session_state.topic_information['topic_uid'],
                pmid_to_human_screening=pmid_to_screening_result,
                source='manual',
            )
            # force refresh of the dataframe
            if 'df' in st.session_state.topic_information:
                del st.session_state.topic_information['df']
    if 'df' not in st.session_state.topic_information or 'article_data_df' not in st.session_state.topic_information:
        count, pmids, article_data_df, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
        st.session_state.topic_information['count'] = count
        st.session_state.topic_information['pmids'] = pmids
        st.session_state.topic_information['article_data_df'] = article_data_df
        st.session_state.topic_information['df'] = df
    else:
        count = st.session_state.topic_information['count']
        pmids = st.session_state.topic_information['pmids']
        article_data_df = st.session_state.topic_information['article_data_df']
        df = st.session_state.topic_information['df']

    e = None


# part two: display pmids
if len(search_text) > 0:
    if st.session_state.topic_information['final'] != 1:
        if count == 0:
            st.markdown('Found no results for this search, generate a new search before committing to this one!')
            st.stop()
        if e is not None:
            st.markdown('Caught an error! Errors in the 500 range might mean the NCBI server is down or struggling')
            st.markdown(e)
            st.stop()
    else:
        if count == 0:
            st.markdown('Found no results for this search, generate a new search before committing to this one!')
            st.stop()

    keep_columns = ['pmid', 'human_decision', 'robot_ranking', 'titles', 'abstracts']
    df = df[keep_columns]
    if st.session_state.topic_information['final'] == 1:
        edit_columns = ['human_decision']
    else:
        st.write('To manually screen, the search strategy must be "finalized" by selecting the button above. At this point, no searches may be modified or added to this topic, and the list of pmids to screen will be frozen.')
        edit_columns = []
    frozen_columns = set(keep_columns) - set(edit_columns)
    if 'screening_results' not in st.session_state.topic_information:
        st.session_state.topic_information['screening_results'] = df

    if st.session_state.topic_information.get('final', 0) == 1:
        if finetune_ranker := st.button('Finetune AutoRanker'):
            with st.spinner('Finetuning the auto ranker and reranking'):
                database_utils.finetune_ranker(st.session_state.topic_information['topic_uid'])
        if st.button('Run AutoRanker (~1 minute / 5k)?') or finetune_ranker:
            database_utils.run_robot_ranker(st.session_state.topic_information['topic_uid'])
            count, pmids, article_data_df, df, e = database_utils.perform_pubmed_search(search_query, st.session_state.topic_information['topic_uid'], persist=1, run_ranker=True, fetch_all_by_date=True)
            st.session_state.topic_information['last_searched'] = search_query
            st.session_state.topic_information['count'] = count
            st.session_state.topic_information['pmids'] = pmids
            st.session_state.topic_information['article_data_df'] = article_data_df
            st.session_state.topic_information['df'] = df

    screening_status = Counter(st.session_state.topic_information.get('screening_results', df)['human_decision'])
    st.markdown(f'Fetched {count} documents. {screening_status["Unscreened"]} Unscreened, {screening_status["Include"]} Include, and {screening_status["Exclude"]} Exclude decisions')
    if 'index' in st.session_state.topic_information.get('df', df).columns:
        del st.session_state.topic_information.get('df', df)['index']


    screening_results = st.data_editor(
        st.session_state.topic_information.get('df', df),
        column_config={
            'pmid': st.column_config.TextColumn(
                'PMID',
                help='pubmed ID',
                width='small',
            ),
            'human_decision': st.column_config.SelectboxColumn(
                'Screening',
                help='Screen in or out this result',
                width='small',
                options=[
                    'Unscreened',
                    'Include',
                    'Exclude'
                ],
            ),
            'robot_ranking': st.column_config.NumberColumn(
                'AutoRank',
                help='AutoRanker Results',
                width='small',
            ),
            'titles': st.column_config.TextColumn(
                'Title',
                help='pubmed article title',
                width='large',
            ),
            'abstracts': st.column_config.TextColumn(
                'Abstract',
                help='pubmed article abstract',
                width='large',
            ),
        },
        # only allow editing the screening decision
        disabled=frozen_columns,
        hide_index=True,
        num_rows='dynamic',
        use_container_width=True,
    )

    if st.session_state.topic_information.get('final', 0) == 1:
        print('saving screening results', Counter(screening_results['human_decision']))
        pmids_pairs = zip(screening_results['pmid'], screening_results['human_decision'])
        # don't bother to insert Unscreened results
        pmids_pairs = filter(lambda x: x[1] != 'Unscreened', pmids_pairs)
        database_utils.insert_topic_human_screening_pubmed_results(
            st.session_state.topic_information['topic_uid'],
            dict(pmids_pairs),
        )
        st.session_state.topic_information['screening_results'] = screening_results


if st.session_state.topic_information.get('final', 0) == 1 and st.button("View Evidence Map"):
    print('saving screening results post submit button', Counter(st.session_state.topic_information['screening_results']['human_decision']))
    database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], dict(zip(st.session_state.topic_information['screening_results']['pmid'], st.session_state.topic_information['screening_results']['human_decision'])))
    st.switch_page('pages/6-evidence_map.py')

if st.button('Start Individual Article Screening'):
    st.switch_page('pages/5-individual_screening.py')

# TODO add a linkout to the pubmed article
# TODO reorder columns for better user experience
