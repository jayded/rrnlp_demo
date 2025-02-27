import time

import streamlit as st

from markupsafe import escape

st.set_page_config(page_title='Search Development', layout='wide')

import rrnlp.app.database_utils as database_utils
#from database_utils import get_next_topic_uid
import rrnlp.models.SearchBot as SearchBot

from rrnlp.app.celery_app import generate_search as celery_generate_search

# manage what page we're on
if 'uid' not in st.session_state:
    st.switch_page('streamlit_app.py')

# this does _not_ check for a topic_uid because we don't initially have one, and don't want to until we insert one into the database
if 'topic_information' not in st.session_state or 'topic_name' not in st.session_state.topic_information:
    st.switch_page('pages/2-existing_projects.py')

if st.session_state.topic_information['final'] == 1:
    st.switch_page('pages/4-search_results_and_screening.py')

# load some streamlit information
# mystery: why does the st module not...seem cooperate?
if not st.session_state.get('loaded_config', False):
    database_utils.load_config()

# speed up loading, so this only happens once.
# for development

@st.cache_data
def page_generate_search(search_text):
    generate_start = time.time()
    print(f'generating search: "{search_text}"')
    query = celery_generate_search(search_text)
    generate_end = time.time()
    print(f'generating the query took {generate_end - generate_start} seconds')
    st.session_state.running_generate_search = False
    return query


print('develop_search')
for x in ['topic_name', 'topic_uid', 'search_text', 'search_query', 'generated_query', 'used_cochrane_filter', 'used_robot_reviewer_rct_filter', 'final']:
    print(x, st.session_state.topic_information.get(x, 'none'))

with st.form("search_form"):

    # generate a search / fill in the default form value with the old / existing one
    old_search_text = st.session_state.topic_information.get('search_text', '').strip()
    new_search_text = st.text_area(f'Enter a description, e.g. a review title, for "{st.session_state.topic_information["topic_name"]}":', value=old_search_text).strip()
    generate_submitted = st.form_submit_button(
        "Generate Search",
        disabled=st.session_state.get('running_generate_search', False),
    )
    if st.session_state.get('running_generate_search', False):
        st.stop()
    # resurrect the old query or generate a new one
    if generate_submitted and len(new_search_text) > 0:
        if old_search_text == new_search_text and len(st.session_state.topic_information.get('generated_query', '')) > 0:
            query = st.session_state.topic_information['search_query']
        else:
            with st.spinner('Generating search'):
                st.markdown('Generating the search may take a moment, now is a good time to fetch a coffee')
                query = page_generate_search(new_search_text)
                st.session_state.topic_information['generated_query'] = query
                st.session_state.topic_information['search_text'] = new_search_text
        #query = '(statins [heart OR cardiovascular] AND ("impact" OR "effect" OR "benefit"))'
        st.session_state.topic_information['search_query'] = query
        st.session_state.topic_information['topic_uid'] = database_utils.write_topic_info(
            topic_uid=st.session_state.topic_information.get('topic_uid', None),
            uid=st.session_state.uid,
            topic_name=st.session_state.topic_information['topic_name'],
            search_text=st.session_state.topic_information['search_text'],
            used_cochrane_filter=st.session_state.topic_information.get('used_cochrane_filter', 0),
            used_robot_reviewer_rct_filter=0,
            search_query=st.session_state.topic_information['search_query'],
            generated_query=st.session_state.topic_information['generated_query'],
            final=0,
        )
    else:
        query = st.session_state.topic_information.get('search_query', None)


if len(st.session_state.topic_information.get('search_text', '')) > 0 and len(str(query)) > 0:
    cochrane_filter = SearchBot.PubmedQueryGeneratorBot.rct_filter()
    with st.form("search"):
        query = st.text_area('Search query', value=st.session_state.topic_information['search_query'])
        if st.checkbox(f'Add Cochrane RCT filter? {escape(cochrane_filter)}', value=st.session_state.topic_information.get('used_cochrane_filter', 0) == 1):
            st.session_state.topic_information['used_cochrane_filter'] = 1
            added_filter = ' AND ' + cochrane_filter
            use_rct_filter = True
        else:
            st.session_state.topic_information['used_cochrane_filter'] = 0
            added_filter = ''
            use_rct_filter = False
        st.session_state.topic_information['run_ranker'] = st.checkbox('Run AutoRanker (~1 minue / 5k)?', value=st.session_state.topic_information.get('run_ranker', False))
        st.session_state.topic_information['search_query'] = query
        execute_submitted = st.form_submit_button("Perform Search")
        if execute_submitted:
            database_utils.write_topic_info(
                topic_uid=st.session_state.topic_information.get('topic_uid', None),
                uid=st.session_state.uid,
                topic_name=st.session_state.topic_information['topic_name'],
                search_text=st.session_state.topic_information['search_text'],
                used_cochrane_filter=st.session_state.topic_information.get('used_cochrane_filter', 0),
                used_robot_reviewer_rct_filter=0,
                search_query=st.session_state.topic_information['search_query'],
                generated_query=st.session_state.topic_information['generated_query'],
                final=st.session_state.topic_information.get('final', 0),
            )
            st.session_state.topic_information['execute_search'] = True
            with st.spinner('Searching! This can take a moment if the search isn\'t very specific'):
                count, pmids, article_data_df, df, e = database_utils.perform_pubmed_search(
                    st.session_state.topic_information['search_query'] + added_filter,
                    st.session_state.topic_information['topic_uid'],
                    persist=False,
                    run_ranker=st.session_state.topic_information['run_ranker'],
                    fetch_all_by_date=True,
                )
                df.set_index('pmid', drop=True, inplace=True)
                st.session_state.topic_information['count'] = count
                st.session_state.topic_information['pmids'] = pmids
                st.session_state.topic_information['article_data_df'] = article_data_df
                st.session_state.topic_information['df'] = df
                st.session_state.topic_information['screening_results'] = df
                st.session_state.topic_information['last_searched'] = st.session_state.topic_information['search_query']
                st.session_state.topic_information['execute_search'] = False
                if count > 0:
                    st.markdown(f'Retrieved {count} documents')
                else:
                    st.markdown(f'Retrieved 0 documents, try a less restrictive search.')

if st.session_state.topic_information.get('df', None) is not None:
    st.markdown(f'Retrieved {len(st.session_state.topic_information["df"])} articles')
    if 'index' in st.session_state.topic_information['df'].columns:
        del st.session_state.topic_information['df']['index']
    if st.button('Finalize search and begin screening?'):
        st.session_state.topic_information['final'] = 1
        finalize = 1
        if 'pmid' in st.session_state.topic_information['df'].columns:
            pmids = st.session_state.topic_information['df']['pmid']
        else:
            pmids = st.session_state.topic_information['df'].index.tolist()

        database_utils.insert_unscreened_pmids(
            topic_uid=st.session_state.topic_information['topic_uid'],
            pmids=pmids,
            ranks=st.session_state.topic_information['df']['robot_ranking'] if 'robot_ranking' in st.session_state.topic_information['df'].columns else None,
        )
        database_utils.write_topic_info(
            topic_uid=st.session_state.topic_information.get('topic_uid', None),
            uid=st.session_state.uid,
            topic_name=st.session_state.topic_information['topic_name'],
            search_text=st.session_state.topic_information['search_text'],
            used_cochrane_filter=st.session_state.topic_information.get('used_cochrane_filter', 0),
            used_robot_reviewer_rct_filter=0,
            search_query=st.session_state.topic_information['search_query'],
            generated_query=st.session_state.topic_information['generated_query'],
            final=1,
        )
        st.switch_page('pages/4-search_results_and_screening.py')
    else:
        finalize = 0
    # TODO add button to rerun the ranker here?
    st.dataframe(
        st.session_state.topic_information['df'],
        hide_index=True,
        use_container_width=True,
        column_config={
            'pmid': st.column_config.TextColumn(
                'PMID',
                help='pubmed ID',
                width='small',
            ),
            # TODO should probably remove this?
            'human_decision': st.column_config.SelectboxColumn(
                'Screening',
                help='Screen in or out this result',
                options=[
                    'Unscreened',
                    'Include',
                    'Exclude'
                ],
                width='small',
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
    )

# TODO add a linkout to the pubmed article
