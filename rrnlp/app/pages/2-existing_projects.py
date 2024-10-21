import pandas as pd
import streamlit as st
st.set_page_config(page_title='Project Management', layout='wide')

import rrnlp.app.database_utils as database_utils
import rrnlp.app.utils as utils

if 'uid' not in st.session_state:
    st.switch_page('streamlit_app.py')

if 'topic_information' in st.session_state:
    del st.session_state['topic_information']

# mystery: why does the st module not...seem cooperate?
if not st.session_state.get('loaded_config', False):
    database_utils.load_config()

st.session_state['topic_information'] = {}

# TODO topic_uid should include something like a creating user id to handle conflicts in a multi-user setting


# TODO allow deletion of topics
# new topic
topic_name = st.text_input('Create a new topic (description here):', st.session_state.topic_information.get('topic_name', ''))
if len(topic_name) > 0:
    st.session_state['topic_information']['topic_name'] = topic_name
    st.session_state['topic_information']['topic_uid'] = database_utils.get_next_topic_uid(st.session_state.uid, topic_name, '', '')
    st.session_state['topic_information']['finalize'] = 0
    st.switch_page('pages/3-develop_search.py')
else:
    st.markdown('Select an existing project with the left checkbox column:')
    current_topic_count = utils.projects_view()
