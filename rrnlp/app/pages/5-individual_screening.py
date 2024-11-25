import itertools
import json
import frozendict
import pandas as pd
import streamlit as st
st.set_page_config(page_title='Search Results and Screening', layout='wide')

import rrnlp.app.database_utils as database_utils

if 'uid' not in st.session_state:
    st.switch_page('streamlit_app.py')
if 'topic_information' not in st.session_state or 'topic_uid' not in st.session_state.topic_information or 'topic_name' not in st.session_state.topic_information or 'final' not in st.session_state.topic_information:
    st.switch_page('pages/2-existing_projects.py')

if st.session_state.topic_information['final'] != 1:
    st.switch_page('pages/3-develop_search.py')

if 'current_screening' not in st.session_state.topic_information:
    st.session_state.topic_information['current_screening'] = {}
if 'screened' not in st.session_state.topic_information['current_screening']:
    st.session_state.topic_information['current_screening']['screened'] = set()

print('individual article screening')
for x in ['topic_name', 'topic_uid', 'search_text', 'search_query', 'generated_query', 'used_cochrane_filter', 'used_robot_reviewer_rct_filter', 'final']:
    print(x, st.session_state.topic_information.get(x, 'none'))

for x in ['current_pmid']:
    print(x, st.session_state.topic_information['current_screening'].get(x, 'none'))


column1, column2 = st.columns([.75, .25])

def insert(pmid, decision):
    database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], {pmid: decision})
    df = st.session_state.topic_information['df']
    # this line is to keep state with the bigger screening dataframe. It's our (local) source of truth.
    df.loc[pmid, 'human_decision'] = decision
    for decision in ['Include', 'Exclude']:
        print(decision, df[df['human_decision'] == decision].index.values.tolist())

# Note to any future developers:
# - the order here is important due to streamlit's execution order.
# - First the page executes, and there have been no decisions
# - then when someone makes a selection, the page re-executes
# - by having all the logic of fetching the next topic and information happen _after_ the button actions execute, we can get the next article ready.

with column1:
    if 'pmids' not in st.session_state.topic_information['current_screening'] or len(st.session_state.topic_information['current_screening']['pmids']) == 0:
        if 'df' not in st.session_state.topic_information or 'article_data_df' not in st.session_state.topic_information:
            _, _, article_data_df, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
            st.session_state.topic_information['article_data_df'] = article_data_df
            st.session_state.topic_information['df'] = df
        df = st.session_state.topic_information['df']
        df_ = df[df['human_decision'] == 'Unscreened']

        to_screen_pmids = df_.index.values.tolist()
        # remove duplicates but retain order
        to_screen_pmids = list({x:x for x in to_screen_pmids}.keys())
        st.session_state.topic_information['current_screening']['pmids'] = to_screen_pmids

    df = st.session_state.topic_information['df']
    st.session_state.topic_information['current_screening']['screened'].update(df[df['human_decision'] != 'Unscreened'].index.values)
    article_data_df = st.session_state.topic_information['article_data_df']

    if 'current_pmid' in st.session_state.topic_information['current_screening'] and st.session_state.topic_information['current_screening']['current_pmid'] in st.session_state.topic_information['current_screening']['screened']:
        del st.session_state.topic_information['current_screening']['current_pmid']

    if 'current_pmid' not in st.session_state.topic_information['current_screening']:
        if len(st.session_state.topic_information['current_screening']['pmids']) > 0:
            st.session_state.topic_information['current_screening']['current_pmid'] = st.session_state.topic_information['current_screening']['pmids'].pop(0)
            print(f"Selected new pmid {st.session_state.topic_information['current_screening']['current_pmid']}")
        else:
            st.markdown('No more pmids to screen! (Review your selections on the bulk screening page)')

            st.stop()

    this_article = article_data_df[article_data_df['pmid'] == st.session_state.topic_information['current_screening']['current_pmid']]
    this_article_data_len = len(this_article)

    while this_article_data_len == 0:
        print(f"Skipping {st.session_state.topic_information['current_screening']['current_pmid']}")

        st.session_state.topic_information['current_screening']['current_pmid'] = st.session_state.topic_information['current_screening']['pmids'].pop(0)
        print(f"Selected new pmid {st.session_state.topic_information['current_screening']['current_pmid']} (last article had no information)")
        this_article = article_data_df[article_data_df['pmid'] == st.session_state.topic_information['current_screening']['current_pmid']]
        this_article_data_len = len(this_article)

    st.write(f"""<a href="https://pubmed.ncbi.nlm.nih.gov/{st.session_state.topic_information['current_screening']['current_pmid']}">PMID{st.session_state.topic_information['current_screening']['current_pmid']}</a>""", unsafe_allow_html=True)

    this_article = article_data_df[article_data_df['pmid'] == st.session_state.topic_information['current_screening']['current_pmid']]
    assert len(this_article) == 1

    st.markdown(this_article['title'].tolist()[0])
    st.markdown(this_article['abstract'].tolist()[0])

    ico_re, pio_df, extractions = database_utils.get_extractions_for_pmids([st.session_state.topic_information['current_screening']['current_pmid']])
    if len(ico_re) > 0:
        # a typo was made
        if 'cvidence' in ico_re.columns:
            del ico_re['evidence']
            ico_re = ico_re.rename(mapper={'cvidence': 'evidence'}, axis=1)
        if 'pmid' in ico_re.columns:
            del ico_re['pmid']
        st.markdown('Study Arms and Measures')
        st.dataframe(ico_re, hide_index=True, use_container_width=True)

    if len(pio_df) > 0:
        col1, col2 = st.columns([.6, .4])
        if 'pmid' in pio_df.columns:
            del pio_df['pmid']
        mesh_rows = pio_df['type'].apply(lambda x: 'mesh' in x)
        pio_mesh = pio_df[mesh_rows]
        pio_extractions = pio_df[~mesh_rows]
        with col1:
            if len(pio_extractions) > 0:
                st.markdown('Participant, Intervention, Outcome extractions')
                del pio_extractions['mesh_term']
                del pio_extractions['mesh_ui']
                del pio_extractions['cui']
                st.dataframe(pio_extractions, hide_index=True, use_container_width=True)
        with col2:
            if len(pio_mesh) > 0:
                st.markdown('PIO MeSH Extractions')
                del pio_mesh['value']
                st.dataframe(pio_mesh, hide_index=True, use_container_width=True)
    if 'pmid' in extractions.columns:
        del extractions['pmid']
    study_df = extractions.melt(var_name='Variable', value_name='Value')
    study_df = study_df.to_dict(orient='records')
    study_df = map(frozendict.frozendict, study_df)
    study_df = set(study_df)
    study_df = pd.DataFrame(study_df).sort_values(by='Variable')
    study_df = study_df.dropna()
    study_df = study_df[study_df['Value'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0 and x != '""')]

    if len(study_df) > 0:
        st.markdown('Study information')
        st.dataframe(
            study_df,
            hide_index=True,
            use_container_width=True,
        )


with column2:
    include = st.button(
        'Include',
        use_container_width=True,
        icon=":material/thumb_up:",
        on_click=lambda: insert(st.session_state.topic_information['current_screening']['current_pmid'], 'Include'),
        key=f"include{st.session_state.topic_information['current_screening']['current_pmid']}",
    )
    exclude = st.button(
        'Exclude',
        use_container_width=True,
        icon=":material/thumb_down:",
        on_click=lambda: insert(st.session_state.topic_information['current_screening']['current_pmid'], 'Exclude'),
        key=f"exclude{st.session_state.topic_information['current_screening']['current_pmid']}",
    )
    skip = st.button(
        'Skip',
        use_container_width=True,
        icon=":material/question_mark:",
        key=f"skip{st.session_state.topic_information['current_screening']['current_pmid']}",
    )

    if include or exclude or skip:
        if include:
            st.session_state.topic_information['counts']['Include'] += 1
            st.session_state.topic_information['counts']['Unscreened'] -= 1
        if exclude:
            st.session_state.topic_information['counts']['Exclude'] += 1
            st.session_state.topic_information['counts']['Unscreened'] -= 1

        st.session_state.topic_information['current_screening']['screened'].add(st.session_state.topic_information['current_screening']['current_pmid'])
        del st.session_state.topic_information['current_screening']['current_pmid']

    if st.button("Return to bulk screening", use_container_width=True):
        st.switch_page('pages/4-search_results_and_screening.py')

    if st.button("View Evidence Map", use_container_width=True):
        st.switch_page('pages/6-evidence_map.py')
