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

if 'topic_information' not in st.session_state or 'topic_uid' not in st.session_state.topic_information or 'topic_name' not in st.session_state.topic_information:
    st.switch_page('pages/2-existing_projects.py')

if st.session_state.topic_information['finalize'] == 1:
    st.switch_page('pages/4-search_results_and_screening.py')

# load some streamlit information
# mystery: why does the st module not...seem cooperate?
if not st.session_state.get('loaded_config', False):
    database_utils.load_config()

# speed up loading, so this only happens once.
# for development
searcher = get_searcher()

@st.cache_data
def generate_search(search_prompt):
    print('getting searcher')
    get_searcher_start = time.time()
    searcher = get_searcher()
    get_searcher_end = time.time()
    print(f'getting the searcher took {get_searcher_end - get_searcher_start} seconds')
    print(f'generating search for prompt {search_prompt}')
    generate_start = time.time()
    query = searcher.generate_review_topic(search_prompt)
    generate_end = time.time()
    print(f'generating the query took {generate_end - generate_start} seconds')
    st.session_state.running_generate_search = False
    return query

def set_generating_search():
    st.session_state.running_generate_search = True

print('develop_search')
for x in ['topic_name', 'topic_uid', 'search_prompt', 'query', 'finalize']:
    print(x, st.session_state.topic_information.get(x, 'none'))

#st.session_state.topic_name = st.text_input('Free text topic description', st.session_state.get('topic_name', ''))
#st.markdown(f'Generate a search for research topic "{st.session_state.topic_information["topic_name"]}"')
#if 'topic_uid' not in st.session_state.topic_information and (
#        len(st.session_state.topic_name) > 0 or
#        'query' in st.session_state
#    ):
#    st.session_state.topic_uid = database_utils.get_next_topic_uid(st.session_state.uid, st.session_state.topic_name, st.session_state.get('search_prompt', ''), st.session_state.get('query', ''))   

#command = 'Translate the following into a Boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Prefer shorter queries.'
#st.markdown(f"Enter a systematic review title or topic, it will be passed to a model with the command \"{command}\"")
# TODO save these session information in some kind of persistence...
#st.session_state.messages = []
with st.form("search_form"):
    command = 'Translate the following into a Boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Prefer shorter queries.'
    st.markdown(f"Enter a systematic review title or topic, it will be passed to a model with the command \"{command}\"")
    old_search_prompt = st.session_state.topic_information.get('search_prompt', '')
    st.session_state.topic_information['search_prompt'] = st.text_area('Enter a description, e.g. a review title, for the topic you\'re interested in:', value=old_search_prompt)
    submitted = st.form_submit_button(
        "Generate Search",
        disabled=st.session_state.get('running_generate_search', False),
        #onclick=set_generating_search,
    )
    if st.session_state.get('running_generate_search', False):
         st.stop()
    if submitted and len(st.session_state['topic_information']['search_prompt']) > 0:
        prompt = st.session_state.topic_information['search_prompt']
        if old_search_prompt == st.session_state.topic_information['search_prompt']:
            query = st.session_state.topic_information['query']
        else:
            with st.spinner('Generating search'):
                query = generate_search(prompt)
        #query = '(statins [heart OR cardiovascular] AND ("impact" OR "effect" OR "benefit"))'
        # save!
        st.session_state.topic_information['query'] = query
        database_utils.write_topic_info(
            topic_uid=st.session_state.topic_information['topic_uid'],
            uid=st.session_state.uid,
            topic_name=st.session_state.topic_information['topic_name'],
            search_prompt=st.session_state.topic_information['search_prompt'],
            query=st.session_state.topic_information['query'],
            final=st.session_state.topic_information['finalize'])
    else:
        query = st.session_state.topic_information.get('query', None)
#    st.session_state.topic_uid = database_utils.get_next_topic_uid(st.session_state.uid, st.session_state.topic_name, st.session_state.get('search_prompt', ''), st.session_state.get('query', ''))   
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#    if 'search_prompt' in st.session_state:
#        st.session_state.messages.append({'role': 'user', 'content': st.session_state.search_prompt})
#        if 'query' in st.session_state:
#            st.session_state.messages.append({'role': 'assistant', 'content': st.session_state.query})
### Display chat messages from history on app rerun
##for message in st.session_state.messages:
##    with st.chat_message(message["role"]):
##        st.markdown(message["content"])
#
##st.text_input("Translate the following into a Boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Prefer shorter queries.", key="name")
### Initialize chat history
#
#query = ''
#for message in st.session_state.messages:
#    role = message['role']
#    content = message['content']
#    with st.chat_message(role):
#        st.markdown(content)
#        if role == 'assistant':
#            query = content
#
#if prompt := st.chat_input("Review topic"):
#    # Display user message in chat message container
#    with st.chat_message("user"):
#        st.markdown(prompt)
#    # Add user message to chat history
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    print('prompt:', prompt)
#    st.session_state.search_prompt = prompt
#    if 'topic_name' not in st.session_state:
#        st.session_state.topic_name = prompt
#    if 'topic_uid' not in st.session_state:
#        st.session_state.topic_uid = database_utils.get_next_topic_uid(st.session_state.uid, st.session_state.topic_name, st.session_state.search_prompt, st.session_state.get('query', ''))   
#
#    # bot response
#    # for development
#    #query = generate_search(prompt)
#    query = '(statins [heart OR cardiovascular] AND ("impact" OR "effect" OR "benefit"))'
#    # I haven't figured out how this creeps in!
#    query = query.replace('&#', '"')
#    query = query.replace('34;', '"')
#    st.session_state.query = query
#    print('query', query)
#    with st.chat_message("assistant"):
#        response = st.markdown(query)
#        database_utils.write_topic_info(st.session_state.topic_uid, st.session_state.uid, st.session_state.topic_name, st.session_state.search_prompt, st.session_state.query)
#    # Add assistant response to chat history
#    st.session_state.messages.append({"role": "assistant", "content": query})
#


# TODO form change on update - how to add the pubmed cochrane filter and process changes?
# TODO also save whether the cochrane filter was added?
# TODO save the edited search separately?
if len(st.session_state.topic_information.get('search_prompt', '')) > 0 and len(str(query)) > 0:
    cochrane_filter = SearchBot.PubmedQueryGeneratorBot.rct_filter()
    if st.checkbox(f'Add Cochrane RCT filter? {cochrane_filter}', value=st.session_state.topic_information.get('use_rct_filter', False)):
        search_query = query + ' AND ' + cochrane_filter
        use_rct_filter = True
    else:
        search_query = query
        use_rct_filter = False
    with st.form("search"):
        searched = st.text_area('Search query', value=search_query)
        submitted = st.form_submit_button("Perform Search")
        if submitted:
            st.session_state.topic_information['searched'] = searched
            st.session_state.topic_information['use_rct_filter'] = use_rct_filter
            st.switch_page('pages/4-search_results_and_screening.py')
