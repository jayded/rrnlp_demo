import pandas as pd
import streamlit as st
st.set_page_config(page_title='Search Results and Screening', layout='wide')

import rrnlp.app.database_utils as database_utils


if 'uid' not in st.session_state:
    st.switch_page('streamlit_app.py')
if 'topic_information' not in st.session_state or 'topic_uid' not in st.session_state.topic_information or 'topic_name' not in st.session_state.topic_information or 'final' not in st.session_state.topic_information:
    st.switch_page('pages/2-existing_projects.py')

if st.session_state.topic_information['final'] != 1:
    st.switch_page('pages/4-search_results_and_screening.py')

# mystery: why does the st module not...seem cooperate?
if not st.session_state.get('loaded_config', False):
    database_utils.load_config()

included_documents_only = st.checkbox('Included Documents Only?', key='IncludedOnly')
all_included_pmids, ico_re, picos, article_data = database_utils.get_auto_evidence_map_from_topic_uid(st.session_state.topic_information['topic_uid'], included_documents_only=included_documents_only)

if 'cvidence' in ico_re.columns:
    del ico_re['evidence']
    ico_re = ico_re.rename(mapper={'cvidence': 'evidence'}, axis=1)

for df in [ico_re, picos, article_data]:
    if 'index' in df:
        del df['index']
if len(article_data) == 0:
    st.markdown('None of the articles selected have automatic extractions')
    st.stop()

if len(all_included_pmids) == 0:
    st.switch_page('pages/4-search_results_and_screening.py')

if len(article_data) == 0:
    st.markdown(f'No article level extractions!')
else:
    st.markdown(f'Results for {len(article_data)} documents')
    st.dataframe(article_data, hide_index=True, use_container_width=True)

if len(ico_re) == 0:
    st.markdown('No extracted study arms or measures')
else:
    st.markdown('Study Arms, Measures, and Findings')
    st.dataframe(ico_re, hide_index=True, use_container_width=True)

if len(picos) == 0:
    st.markdown('No extracted PIO information')
else:
    st.markdown('Participant, Intervention, Outcome extractions')
    st.dataframe(picos, hide_index=True)
