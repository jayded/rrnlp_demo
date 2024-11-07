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
    df.loc[df['pmid'] == pmid, 'human_decision'] = decision

# Note to any future developers:
# - the order here is important due to streamlit's execution order.
# - First the page executes, and there have been no decisions
# - then when someone makes a selection, the page re-executes
# - by having all the logic of fetching the next topic and information happen _after_ the button actions execute, we can get the next article ready.
with column2:
    include = st.button(
        'Include',
        use_container_width=True,
        icon=":material/thumb_up:",
        on_click=lambda: insert(st.session_state.topic_information['current_screening']['current_pmid'], 'Include'),
    )
    exclude = st.button(
        'Exclude',
        use_container_width=True,
        icon=":material/thumb_down:",
        on_click=lambda: insert(st.session_state.topic_information['current_screening']['current_pmid'], 'Exclude'),
    )
    skip = st.button(
        'Skip',
        use_container_width=True,
        icon=":material/question_mark:",
    )

    if include or exclude or skip:
        st.session_state.topic_information['current_screening']['screened'].add(st.session_state.topic_information['current_screening']['current_pmid'])
        del st.session_state.topic_information['current_screening']['current_pmid']

    if st.button("Return to bulk screening", use_container_width=True):
        st.switch_page('pages/4-search_results_and_screening.py')

    if st.button("View Evidence Map", use_container_width=True):
        st.switch_page('pages/6-evidence_map.py')

with column1:
    if 'pmids' not in st.session_state.topic_information['current_screening'] or len(st.session_state.topic_information['current_screening']['pmids']) == 0:
        if 'df' not in st.session_state.topic_information or 'article_data_df' not in st.session_state.topic_information:
            _, _, article_data_df, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
            st.session_state.topic_information['article_data_df'] = article_data_df
            st.session_state.topic_information['df'] = df
        df = st.session_state.topic_information['df']
        df_ = df[df['human_decision'] == 'Unscreened']

        to_screen_pmids = df_['pmid'].tolist()
        # remove duplicates but retain order
        to_screen_pmids = list({x:x for x in to_screen_pmids}.keys())
        st.session_state.topic_information['current_screening']['pmids'] = to_screen_pmids

    df = st.session_state.topic_information['df']
    st.session_state.topic_information['current_screening']['screened'].update(df[df['human_decision'] != 'Unscreened']['pmid'])
    article_data_df = st.session_state.topic_information['article_data_df']

    if 'current_pmid' in st.session_state.topic_information['current_screening'] and st.session_state.topic_information['current_screening']['current_pmid'] in st.session_state.topic_information['current_screening']['screened']:
        del st.session_state.topic_information['current_screening']['current_pmid']

    if 'current_pmid' not in st.session_state.topic_information['current_screening']:
        if len(st.session_state.topic_information['current_screening']['pmids']) > 0:
            st.session_state.topic_information['current_screening']['current_pmid'] = st.session_state.topic_information['current_screening']['pmids'].pop()
        else:
            st.markdown('No more pmids to screen!')

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

    # TODO should this look like a pubmed page?
    st.markdown(this_article['titles'].tolist()[0])
    st.markdown(this_article['abstracts'].tolist()[0])

    #evidence_map = database_utils.get_auto_evidence_map_from_pmids([st.session_state.topic_information['current_screening']['current_pmid']], include_only=False)
    ico_re, extractions = database_utils.get_extractions_for_pmids([st.session_state.topic_information['current_screening']['current_pmid']])
    if len(ico_re) > 0:
        st.dataframe(ico_re)
    if len(extractions) > 0:
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
        study_df['Value'] = study_df['Value'].apply(lambda x: f"{x:,.3f}" if isinstance(x, float) else str(x))
        study_df = study_df.to_dict(orient='records')
        study_df = map(frozendict.frozendict, study_df)
        study_df = set(study_df)
        study_df = pd.DataFrame(study_df).sort_values(by='Variable')

        st.dataframe(
            study_df,
            hide_index=True,
            use_container_width=True,
        )


