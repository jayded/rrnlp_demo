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
search_prompt = st.session_state.topic_information.get('search_prompt', '')
query = st.session_state.topic_information.get('query', '')

if st.session_state.topic_information['finalize'] != 1:
    with st.form("search"):
        searched = st.text_area('Boolean query', value=search_query)
        submitted = st.form_submit_button("Search")
        if submitted:
            st.session_state.searched = searched

    if len(search_prompt) > 0 and st.checkbox(f'Add Cochrane RCT filter? {cochrane_filter}', value=st.session_state.topic_information.get('use_rct_filter', False)):
        search_query = query + ' AND ' + cochrane_filter
        use_rct_filter = True
    elif len(search_prompt) > 0:
        search_query = query
        use_rct_filter = False
    else:
        search_query = ''
        use_rct_filter = False
    # TODO persist this
    st.session_state.topic_information['use_rct_filter'] = use_rct_filter

    run_ranker = st.checkbox('Run AutoRanker (~1 minute / 5k)?', value=st.session_state.topic_information.get('run_ranker', False))

    st.write('Click "Finalize" to finalize this search and begin screening. Once the search is finalized, no new searches may be added')
    if st.button('Finalize'):
        st.session_state.topic_information['finalize'] = 1
        finalize = 1
    else:
        finalize = 0

    # TODO don't rerun if the search is unchanged
    if search_query != st.session_state.topic_information.get('last_searched', ''):
        count, pmids, article_data, df, e = database_utils.perform_pubmed_search(search_query, st.session_state.topic_information['topic_uid'], persist=1, run_ranker=True)
        #count, pmids, article_data, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
        st.session_state.topic_information['count'] = count
        st.session_state.topic_information['pmids'] = pmids
        st.session_state.topic_information['article_data'] = article_data
        st.session_state.topic_information['df'] = df
        st.session_state.topic_information['screening_results'] = df
        st.session_state.topic_information['last_searched'] = search_query
    else:
        count = st.session_state.topic_information['count']
        pmids = st.session_state.topic_information['pmids']
        article_data = st.session_state.topic_information['article_data']
        df = st.session_state.topic_information['df']
    # we separate these two so the search will be persisted and *then* the topic information updated; doing it the other way means any issues 
    if finalize == 1:
        st.markdown('All pmids from this search will be inserted and persisted into the database')
        database_utils.write_topic_info(
            topic_uid=st.session_state.topic_information['topic_uid'],
            uid=st.session_state.uid,
            topic_name=st.session_state.topic_information['topic_name'],
            search_prompt=st.session_state.topic_information['search_prompt'],
            query=st.session_state.topic_information['query'],
            final=st.session_state.topic_information['finalize'])
else:
    # TODO persist if the cochrane filter got used
    if len(search_prompt) > 0 and st.session_state.topic_information.get('use_rct_filter', False):
        search_query = query + ' AND ' + cochrane_filter
    else:
        search_query = query
    st.markdown(f'Results for: {search_query}')
    st.write('No more search results can be added via pubmed searches. Add any others manually:')
    with st.form('Insert bulk screening results'):
        st.markdown('Insert a list of pmids to Include. Use spaces or commas to separate them')
        include_pmids = st.text_area('Include pmids', value='', height=20)
        st.markdown('Insert a list of pmids to Exclude. Use spaces or commas to separate them')
        exclude_pmids = st.text_area('Exclude pmids', value='', height=20)
        submitted = st.form_submit_button("Insert")
        if submitted:
            include_pmids = list(map(str.split(','), include_pmids.split()))
            exclude_pmids = list(map(str.split(','), exclude_pmids.split()))
            pmid_to_screening_result = {pmid: 'Include' for pmid in include_pmids}
            pmid_to_screening_result.update({pmid: 'Exclude' for pmid in exclude_pmids})
            database_utils.insert_topic_human_screening_pubmed_results(
                topic_uid=st.session_state.topic_information['topic_uid'],
                pmid_to_human_screening=pmid_to_screening_result,
                source='manual',
            )
    if 'df' not in st.session_state.topic_information:
        count, pmids, article_data, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
        #st.session_state.topic_information['last_searched'] = search_query
        st.session_state.topic_information['count'] = count
        st.session_state.topic_information['pmids'] = pmids
        st.session_state.topic_information['article_data'] = article_data
        st.session_state.topic_information['df'] = df
    else:
        #search_query = st.session_state.topic_information['last_searched']
        count = st.session_state.topic_information['count']
        pmids = st.session_state.topic_information['pmids']
        article_data = st.session_state.topic_information['article_data']
        df = st.session_state.topic_information['df']



if len(search_prompt) > 0:
    if st.session_state.topic_information['finalize'] != 1:
        if count == 0:
            st.markdown('Found no results for this search, generate a new search before committing to this one!')
            st.stop()
        if e is not None:
            st.markdown('Caught an error! Errors in the 500 range might mean the NCBI server is down or struggling')
            st.markdown(e)
            st.stop()
    else:
        #count, pmids, article_data, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
        if count == 0:
            st.markdown('Found no results for this search, generate a new search before committing to this one!')
            st.stop()

    keep_columns = ['pmid', 'human_decision', 'robot_ranking', 'titles', 'abstracts']
    df = df[keep_columns]
    print('loaded screening results', Counter(df['human_decision']))
    if st.session_state.topic_information['finalize'] == 1:
        edit_columns = ['human_decision']
    else:
        st.write('To manually screen, the search strategy must be "finalized" by selecting the button above. At this point, no searches may be modified or added to this topic, and the list of pmids to screen will be frozen.')
        edit_columns = []
    frozen_columns = set(keep_columns) - set(edit_columns)
    if 'screening_results' not in st.session_state.topic_information:
        st.session_state.topic_information['screening_results'] = df

    if st.session_state.topic_information.get('finalize', 0) == 1:
        if finetune_ranker := st.button('Finetune AutoRanker'):
            pass
        if st.button('Run AutoRanker (~1 minute / 5k)?') or finetune_ranker:
            database_utils.run_robot_ranker(st.session_state.topic_information['topic_uid'])
            count, pmids, article_data, df, e = database_utils.perform_pubmed_search(search_query, st.session_state.topic_information['topic_uid'], persist=1, run_ranker=True)
            st.session_state.topic_information['last_searched'] = search_query
            st.session_state.topic_information['count'] = count
            st.session_state.topic_information['pmids'] = pmids
            st.session_state.topic_information['article_data'] = article_data
            st.session_state.topic_information['df'] = df
            #database_utils.run_robot_ranker(st.session_state.topic_information['topic_uid'])
    st.session_state.topic_information['screening_results'] = st.data_editor(
        #df,
        st.session_state.topic_information.get('screening_results', df),
        column_config={
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
        #on_change=on_change,
    )

    if st.session_state.topic_information.get('finalize', 0) == 1:
        print('saving screening results', Counter(st.session_state.topic_information['screening_results']['human_decision']))
        database_utils.insert_topic_human_screening_pubmed_results(
            st.session_state.topic_information['topic_uid'],
            dict(
                zip(
                    st.session_state.topic_information['screening_results']['pmid'],
                    st.session_state.topic_information['screening_results']['human_decision']
                )
            )
        )


if st.session_state.topic_information.get('finalize', 0) == 1 and st.button("View Evidence Map"):
    print('saving screening results post submit button', Counter(st.session_state.topic_information['screening_results']['human_decision']))
    database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], dict(zip(st.session_state.topic_information['screening_results']['pmid'], st.session_state.topic_information['screening_results']['human_decision'])))
    st.switch_page('pages/5-evidence_map.py')
