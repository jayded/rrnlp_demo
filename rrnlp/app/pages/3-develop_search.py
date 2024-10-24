import time
import streamlit as st
st.set_page_config(page_title='Search Development', layout='wide')

from rrnlp.app.utils import get_searcher
import rrnlp.app.database_utils as database_utils
#from database_utils import get_next_topic_uid
import rrnlp.models.SearchBot as SearchBot

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
get_searcher_start = time.time()
searcher = get_searcher()
get_searcher_end = time.time()
print(f'getting the searcher took {get_searcher_end - get_searcher_start} seconds')

@st.cache_data
def generate_search(search_text):
    print('getting searcher')
    get_searcher_start = time.time()
    searcher = get_searcher()
    get_searcher_end = time.time()
    print(f'getting the searcher took {get_searcher_end - get_searcher_start} seconds')
    print(f'generating search for prompt {search_text}')
    generate_start = time.time()
    query = searcher.generate_review_topic(search_text)
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
    new_search_text = st.text_area('Enter a description, e.g. a review title, for the topic you\'re interested in:', value=old_search_text).strip()
    generate_submitted = st.form_submit_button(
        "Generate Search",
        disabled=st.session_state.get('running_generate_search', False),
    )
    if st.session_state.get('running_generate_search', False):
        st.stop()
    # resurrect the old query or generate a new one
    if generate_submitted and len(new_search_text) > 0:
        if old_search_text == new_search_text:
            query = st.session_state.topic_information['search_query']
        else:
            with st.spinner('Generating search'):
                st.markdown('Generating the search may take a moment, now is a good time to fetch a coffee')
                query = generate_search(new_search_text)
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
            final=st.session_state.topic_information['final'],
        )
    else:
        query = st.session_state.topic_information.get('search_query', None)


if len(st.session_state.topic_information.get('search_text', '')) > 0 and len(str(query)) > 0:
    cochrane_filter = SearchBot.PubmedQueryGeneratorBot.rct_filter()
    if st.checkbox(f'Add Cochrane RCT filter? {cochrane_filter}', value=st.session_state.topic_information.get('used_cochrane_filter', 0) == 1):
        st.session_state.topic_information['used_cochrane_filter'] = 1
    else:
        st.session_state.topic_information['used_cochrane_filter'] = 0
    st.session_state.topic_information['run_ranker'] = st.checkbox('Run AutoRanker (~1 minue / 5k)?', value=st.session_state.topic_information.get('run_ranker', False))
    with st.form("search"):
        query = st.text_area('Search query', value=st.session_state.topic_information['search_query'])
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
                final=st.session_state.topic_information['final'],
            )
            st.session_state.topic_information['execute_search'] = True
            st.switch_page('pages/4-search_results_and_screening.py')

# TODO add a linkout to the pubmed article
