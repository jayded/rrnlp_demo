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

all_included_pmids, evidence_map = database_utils.get_auto_evidence_map_from_topic_uid(st.session_state.topic_information['topic_uid'])
if len(evidence_map) == 0:
    st.markdown('None of the articles selected have automatic extractions')
    st.stop()

if len(all_included_pmids) == 0:
    st.switch_page('pages/4-search_results_and_screening.py')
evidence_map = pd.DataFrame.from_records(evidence_map)
st.markdown(f'Of {len(all_included_pmids)}, have extractions for {len(set(evidence_map["pmid"]))}, missing extractions for {len(all_included_pmids) - len(set(evidence_map["pmid"]))}')
print(evidence_map.columns, len(evidence_map))


keep_column = ['pmid', 'title', 'abstract', 'intervention', 'comparator', 'outcome', 'label', 'evidence', 'population', 'sample_size', 'prob_low_rob', 'low_rsg_bias', 'low_ac_bias', 'low_bpp_bias']
disabled_columns = set(evidence_map.columns) - set(['human_decision'])
# TODO consider allowing updating the screened columns
# TODO consider allowing processing of the other documents

edited_df = st.data_editor(
    evidence_map, 
    column_config={
        'human_decision': st.column_config.SelectboxColumn(
            'Screen',
            help='Screen in or out this result',
            width='medium',
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
        'intervention': st.column_config.TextColumn(
            'Intervention',
            help='Automatically extracted intervention',
            width='medium',
        ),
        'comparator': st.column_config.TextColumn(
            'Comprator',
            help='Automatically extracted intervention (comparator)',
            width='medium',
        ),
        'outcome': st.column_config.TextColumn(
            'Outcome',
            help='Automatically extracted outcome or endpoint',
            width='medium',
        ),
        #'label': st.column_config.TextColumn(
        #    'Label',
        #    help='Automatically classified statistical finding (sig +/-/~)',
        #    width='small',
        #),
        'evidence': st.column_config.TextColumn(
            'Evidence',
            help='Supporting text from the article (perhaps just abstract) for the Label',
            width='medium',
        ),
        'p': st.column_config.TextColumn(
            'Population',
            help='Automatically extracted population information',
            width='medium',
        ),
    },
    # only allow editing the screening decision
    disabled=disabled_columns,
    hide_index=True,
    num_rows='dynamic',
    #on_change=lambda: database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], dict(zip(edited_df['pmid'], edited_df['human_decision']))),
)
print(edited_df.columns)
# just let the database handle changes (or lack thereof)
#database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], dict(zip(edited_df['pmid'], edited_df['human_decision'])))
