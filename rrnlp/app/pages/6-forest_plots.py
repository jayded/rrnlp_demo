import streamlit as st
#st.set_page_config(page_title='Search Results and Screening', layout='wide')
import rrnlp.app.database_utils as database_utils
import rrnlp.app.utils as utils

if 'uid' not in st.session_state:
    st.switch_page('streamlit_app.py')
if 'topic_information' not in st.session_state or 'topic_uid' not in st.session_state.topic_information or 'topic_name' not in st.session_state.topic_information or 'finalize' not in st.session_state.topic_information:
    st.switch_page('pages/2-existing_projects.py')

if st.session_state.topic_information['finalize'] != 1:
    st.switch_page('pages/4-search_results_and_screening.py')

# mystery: why does the st module not...seem cooperate?
if not st.session_state.get('loaded_config', False):
    database_utils.load_config()

df = utils.extract_numerical_information(st.session_state.topic_information['topic_uid'])
st.dataframe(df)
