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
    st.switch_page('pages/4-search_results_and_screening.py')

if 'current_screening' not in st.session_state.topic_information:
    st.session_state.topic_information['current_screening'] = {}

if 'pmids' not in st.session_state.topic_information['current_screening'] or len(st.session_state.topic_information['current_screening']['pmids']) == 0:
    if 'df' not in st.session_state.topic_information or 'article_data_df' not in st.session_state.topic_information:
        _, _, article_data_df, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
        st.session_state.topic_information['article_data_df'] = article_data_df
        st.session_state.topic_information['df'] = df
    df = st.session_state.topic_information['df']
    df = df[df['human_decision'] == 'Unscreened']
    st.session_state.topic_information['current_screening']['pmids'] = df['pmid'].tolist()

article_data_df = st.session_state.topic_information['article_data_df']

if 'current_pmid' not in st.session_state.topic_information['current_screening']:
    if len(st.session_state.topic_information['current_screening']['pmids']) > 0:
        st.session_state.topic_information['current_screening']['current_pmid'] = st.session_state.topic_information['current_screening']['pmids'].pop()
    else:
        st.markdown('No more pmids to screen!')
        if st.button("Return to bulk screening", use_container_width=True):
            st.switch_page('pages/4-search_results_and_screening.py', use_container_width=True)

        if st.button("View Evidence Map", use_container_width=True):
            st.switch_page('pages/6-evidence_map.py')
        st.stop()

this_article = article_data_df[article_data_df['pmid'] == st.session_state.topic_information['current_screening']['current_pmid']]
this_article_data_len = len(this_article)

while this_article_data_len == 0:
    print(f"Skipping {st.session_state.topic_information['current_screening']['current_pmid']}")

    st.session_state.topic_information['current_screening']['current_pmid'] = st.session_state.topic_information['current_screening']['pmids'].pop()
    this_article = article_data_df[article_data_df['pmid'] == st.session_state.topic_information['current_screening']['current_pmid']]
    this_article_data_len = len(this_article)

st.write(f"""<a href="https://pubmed.ncbi.nlm.nih.gov/{st.session_state.topic_information['current_screening']['current_pmid']}">PMID{st.session_state.topic_information['current_screening']['current_pmid']}</a>""", unsafe_allow_html=True)

this_article = article_data_df[article_data_df['pmid'] == st.session_state.topic_information['current_screening']['current_pmid']]
assert len(this_article) == 1

column1, column2 = st.columns([.75, .25])

with column1:
    # TODO should this look like a pubmed page?
    st.markdown(this_article['titles'].tolist()[0])
    st.markdown(this_article['abstracts'].tolist()[0])

    #evidence_map = database_utils.get_auto_evidence_map_from_pmids([st.session_state.topic_information['current_screening']['current_pmid']], include_only=False)
    ico_re, extractions = database_utils.get_extractions_for_pmids([st.session_state.topic_information['current_screening']['current_pmid']])
    if len(ico_re) > 0:
        st.dataframe(ico_re)
    if len(extractions) > 0:
        # TODO unroll the json in each of these elements
        pio_df = extractions[['p', 'i', 'o']].melt(var_name='Variable', value_name='Value')
        pio_df_rows = pio_df.to_dict(orient='records')
        pio_df_rows = filter(lambda row: row['Value'] is not None, pio_df_rows)
        pio_df_rows = map(lambda x: ({'Variable': x['Variable'], 'Value': v.strip()} for v in json.loads(x['Value'])), pio_df_rows)
        pio_df_rows = itertools.chain.from_iterable(pio_df_rows)
        pio_df_rows = map(frozendict.frozendict, pio_df_rows)
        pio_df_rows = set(pio_df_rows)
        pio_df = pd.DataFrame(pio_df_rows)
        if len(pio_df) > 0:
            st.dataframe(pio_df.sort_values(by=['Variable', 'Value']), hide_index=True, use_container_width=True)
        # TODO delete any empty columns here?
        study_df = extractions[['prob_rct', 'is_rct_sensitive', 'is_rct_balanced', 'is_rct_precise', 'prob_lob_rob', 'num_randomized', 'study_design', 'prob_sr', 'is_sr', 'prob_cohort', 'is_cohort', 'prob_consensus', 'is_consensus', 'prob_ct', 'is_ct', 'prob_ct_protocol', 'is_ct_protocol', 'prob_guideline', 'is_guideline', 'prob_qual', 'is_qual']].melt(var_name='Variable', value_name='Value')
        #study_df = pd.wide_to_long(extractions[['prob_rct', 'is_rct_sensitive', 'is_rct_balanced', 'is_rct_precise', 'prob_lob_rob', 'num_randomized', 'study_design', 'prob_sr', 'is_sr', 'prob_cohort', 'is_cohort', 'prob_consensus', 'is_consensus', 'prob_ct', 'is_ct', 'prob_ct_protocol', 'is_ct_protocol', 'prob_guideline', 'is_guideline', 'prob_qual', 'is_qual']])
        study_df['Value'] = study_df['Value'].apply(lambda x: f"{x:,.3f}" if isinstance(x, float) else str(x))
        study_df = pd.DataFrame(study_df)

        st.dataframe(
            study_df,
            hide_index=True,
            use_container_width=True,
        )


with column2:
    # TODO pretty the buttons
    include = st.button('Include', use_container_width=True)
    exclude = st.button('Exclude', use_container_width=True)
    skip = st.button('Skip', use_container_width=True)

    if include:
        decision = 'Include'
        assert not exclude
    elif exclude:
        decision = 'Exclude'

    if include or exclude:
        database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], {st.session_state.topic_information['current_screening']['current_pmid']: decision})
        df = st.session_state.topic_information.get('df', None)
        # this line is to keep state with the bigger screening dataframe. It's our (local) source of truth.
        if df is not None:
            df[df['pmid'] == st.session_state.topic_information['current_screening']['current_pmid']]['human_decision'] = decision
        del st.session_state.topic_information['current_screening']['current_pmid']

    if st.button("Return to bulk screening", use_container_width=True):
        st.switch_page('pages/4-search_results_and_screening.py')

    if st.button("View Evidence Map", use_container_width=True):
        st.switch_page('pages/6-evidence_map.py')