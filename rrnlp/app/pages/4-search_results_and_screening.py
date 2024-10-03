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

#if 'db_path' not in st.session_state:
#    with open(config_file, 'r', encoding='utf-8') as file:
#        config = yaml.load(file, Loader=SafeLoader)
#    st.session_state.db_path = config['db_path']
#    st.session_state.pubmed_db_path = config['pubmed_db_path']


cochrane_filter = SearchBot.PubmedQueryGeneratorBot.rct_filter()

# TODO don't use the copies when they can be avoided
search_prompt = st.session_state.topic_information.get('search_prompt', '')
query = st.session_state.topic_information.get('query', '')


# TODO set this up so it shows the final query when the search has been finalized
if st.session_state.topic_information['finalize'] != 1:
    if len(search_prompt) > 0 and st.checkbox(f'Add Cochrane RCT filter? {cochrane_filter}', value=st.session_state.topic_information.get('use_rct_filter', False)):
        search_query = query + ' AND ' + cochrane_filter
        use_rct_filter = True
    else:
        use_rct_filter = False

    # TODO persist this
    st.session_state.use_rct_filter = use_rct_filter
    with st.form("search"):
        searched = st.text_area('Boolean query', value=search_query)
        submitted = st.form_submit_button("Search")
        if submitted:
            st.session_state.searched = searched
            #st.switch_page('pages/3-search_results_and_screening.py')
    st.write('Click "Finalize" to finalize this search and begin screening. Once the search is finalized, no new searches may be added')
    if st.button('Finalize'):
        st.session_state.topic_information['finalize'] = 1
        finalize = 1
    else:
        finalize = 0
    count, pmids, article_data, df, e = database_utils.perform_pubmed_search(search_prompt, st.session_state.topic_information['topic_uid'], persist=finalize)
    # we separate these two so the search will be persisted and *then* the topic information updated; doing it the other way means any issues 
    if finalize == 1:
        st.markdown('All new pmids will be inserted and persisted into the database for this particular search')
        database_utils.write_topic_info(
            topic_uid=st.session_state.topic_information['topic_uid'],
            uid=st.session_state.uid,
            topic_name=st.session_state.topic_information['topic_name'],
            search_prompt=st.session_state.topic_information['search_prompt'],
            query=st.session_state.topic_information['query'],
            final=st.session_state.topic_information['finalize'])
else:
    # TODO persist if the cochrane filter got used
    search_query = query
    #if len(search_prompt) > 0 and st.checkbox(f'Add Cochrane RCT filter? {cochrane_filter}', value=st.session_state.topic_information.get('use_rct_filter', False)):
    #    search_query = query + ' AND ' + cochrane_filter
    #else:
    #    search_query = query
    st.markdown(f'Results for: {search_query}')
    st.write('No more search results can be added via pubmed searches. Add any others manually.')

#if st.session_state.topic_information['finalize'] != 1:
#    st.write('Click "Finalize" to finalize this search and begin screening. Once the search is finalized, no new searches may be added')
#    if st.button('Finalize'):
#        st.session_state.topic_information['finalize'] = 1
#        finalize = 1
#    else:
#        finalize = 0
#    count, pmids, article_data, df, e = database_utils.perform_pubmed_search(search_prompt, st.session_state.topic_information['topic_uid'], persist=finalize)
#    # we separate these two so the search will be persisted and *then* the topic information updated; doing it the other way means any issues 
#    if finalize == 1:
#        st.markdown('All new pmids will be inserted and persisted into the database for this particular search')
#        database_utils.write_topic_info(
#            topic_uid=st.session_state.topic_information['topic_uid'],
#            uid=st.session_state.uid,
#            topic_name=st.session_state.topic_information['topic_name'],
#            search_prompt=st.session_state.topic_information['search_prompt'],
#            query=st.session_state.topic_information['query'],
#            final=st.session_state.topic_information['finalize'])
#else:
#    st.write('No more search results can be added via pubmed searches. Add any others manually.')


if len(search_prompt) > 0:
    if st.session_state.topic_information['finalize'] != 1:
        count, pmids, article_data, df, e = database_utils.perform_pubmed_search(search_prompt, st.session_state.topic_information['topic_uid'], persist=st.session_state.topic_information['finalize'] == 1)
        st.session_state.topic_information['pmids'] = pmids
        st.session_state.topic_information['article_data_df'] = df
        #print('dataframe columns', df.columns)
        if count == 0:
            st.markdown('Found no results for this search, generate a new search before committing to this one!')
            st.stop()
        if e is not None:
            st.markdown('Caught an error! Errors in the 500 range might mean the NCBI server is down or struggling')
            st.markdown(e)
            st.stop()
    else:
        count, pmids, article_data, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
        if count == 0:
            st.markdown('Found no results for this search, generate a new search before committing to this one!')
            st.stop()

    keep_columns = ['pmid', 'human_decision', 'robot_ranking', 'title', 'abstract', 'keywords', 'mesh_terms', 'authors']
    df = df[keep_columns]
    print('loaded screening results', Counter(df['human_decision']))
    if st.session_state.topic_information['finalize'] == 1:
        edit_columns = ['human_decision']
    else:
        st.write('To manually screen, the search strategy must be "finalized" by selecting the button above. At this point, no searches may be modified or added to this topic, and the list of pmids to screen will be frozen.')
        # TODO allow manual addition of pmids
        edit_columns = []
    frozen_columns = set(keep_columns) - set(edit_columns)
    # TODO this should only update the screened values
    st.session_state.topic_information['screening_results'] = df
    #if st.session_state.topic_information.get('finalize', 0) == 1:
    #    on_change = lambda: database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], dict(zip(edited_df['pmid'], edited_df['human_decision'])))
    #else:
    #    on_change = None
    edited_df = st.data_editor(
        df, 
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
            'title': st.column_config.TextColumn(
                'Title',
                help='pubmed article title',
                width='large',
            ),
            'abstract': st.column_config.TextColumn(
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
    # just let the database handle changes (or lack thereof)
    if st.session_state.topic_information.get('finalize', 0) == 1:
        print('saving screening results', Counter(edited_df['human_decision']))
        database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], dict(zip(edited_df['pmid'], edited_df['human_decision'])))
        if st.button('Rerun AutoRanker'):
            database_utils.run_robot_ranker(st.session_state.topic_information['topic_uid'])


## TODO train robot screener
#with st.form("Train screener"):
#    submitted = st.form_submit_button("Search")
#    if submitted:
#        st.switch_page('pages/4-search_results_and_screening.py')

with st.form("View Evidence Map"):
    submitted = st.form_submit_button("View Evidence Map")
    if submitted:
        print('saving screening results post submit button', Counter(edited_df['human_decision']))
        database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], dict(zip(edited_df['pmid'], edited_df['human_decision'])))
        st.switch_page('pages/5-evidence_map.py')
