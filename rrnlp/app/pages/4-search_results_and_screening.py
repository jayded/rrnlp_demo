import itertools

from collections import Counter

import pandas as pd
import streamlit as st
st.set_page_config(page_title='Search Results and Screening', layout='wide')

import rrnlp.models.SearchBot as SearchBot

import rrnlp.app.database_utils as database_utils

if 'uid' not in st.session_state:
    st.switch_page('streamlit_app.py')

if 'topic_information' not in st.session_state or 'topic_uid' not in st.session_state.topic_information or 'topic_name' not in st.session_state.topic_information:
    st.switch_page('pages/2-existing_projects.py')

if st.session_state.topic_information['final'] != 1:
    st.switch_page('pages/3-develop_search.py')

# mystery: why does the st module not...seem cooperate?
if not st.session_state.get('loaded_config', False):
    database_utils.load_config()

print('bulk screening')
for x in ['topic_name', 'topic_uid', 'search_text', 'search_query', 'generated_query', 'used_cochrane_filter', 'used_robot_reviewer_rct_filter', 'final']:
    print(x, st.session_state.topic_information.get(x, 'none'))

cochrane_filter = SearchBot.PubmedQueryGeneratorBot.rct_filter()

# TODO don't use the copies when they can be avoided
search_query = st.session_state.topic_information['search_query']
st.markdown(f'Results for: {search_query}' + (f' AND <pre>{cochrane_filter}</pre>' if st.session_state.topic_information.get('used_cochrane_filter', 0) == 1 else ''))
st.write('No more search results can be added via pubmed searches. Add any others manually:')
with st.form('Insert bulk screening results'):
    st.markdown('Insert a list of pmids to Include. Use spaces or commas to separate them')
    include_pmids = st.text_area('Include pmids', value='', height=20)
    st.markdown('Insert a list of pmids to Exclude. Use spaces or commas to separate them')
    exclude_pmids = st.text_area('Exclude pmids', value='', height=20)
    submitted = st.form_submit_button("Insert")
    if submitted:
        include_pmids = set(itertools.chain.from_iterable([map(str.strip, x.split(',')) for x in include_pmids.split()]))
        exclude_pmids = set(itertools.chain.from_iterable([map(str.strip, x.split(',')) for x in exclude_pmids.split()]))
        # TODO rerun screening if there were any before?
        if len(include_pmids & exclude_pmids) > 0:
            st.markdown(f'Warning: no ids inserted, you have duplicates: {include_pmids & exclude_pmids}')
        else:
            pmid_to_screening_result = {**{pmid: 'Include' for pmid in include_pmids}, **{pmid: 'Exclude' for pmid in exclude_pmids}}
            database_utils.insert_topic_human_screening_pubmed_results(
                topic_uid=st.session_state.topic_information['topic_uid'],
                pmid_to_human_screening=pmid_to_screening_result,
                source='manual',
            )
            # force refresh of the dataframe
            if 'df' in st.session_state.topic_information:
                del st.session_state.topic_information['df']
            if 'article_data_df' in st.session_state.topic_information:
                del st.session_state.topic_information['article_data_df']
            if 'screening_results' in st.session_state.topic_information:
                del st.session_state.topic_information['screening_results']



# TODO fix the guardrails
# TODO somehow screening results aren't saved consistently on this side?
counts = st.session_state.topic_information.get('counts', None)
if counts is None:
    counts = Counter(st.session_state.topic_information.get('screening_results', df)['human_decision'])
    st.session_state.topic_information['counts'] = counts

if finetune_ranker := st.button('Finetune AutoRanker', disabled=counts.get('Include', 0) == 0):
    with st.spinner('Finetuning the auto ranker and reranking (time for another coffee)'):
        database_utils.finetune_ranker(st.session_state.topic_information['topic_uid'])
if st.button('Run AutoRanker (~1 minute / 5k)?') or finetune_ranker:
    database_utils.run_robot_ranker(st.session_state.topic_information['topic_uid'])

if 'df' not in st.session_state.topic_information or 'article_data_df' not in st.session_state.topic_information:
    count, pmids, article_data_df, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
    st.session_state.topic_information['count'] = count
    st.session_state.topic_information['pmids'] = pmids
    st.session_state.topic_information['article_data_df'] = article_data_df
    st.session_state.topic_information['df'] = df
else:
    count = st.session_state.topic_information['count']
    pmids = st.session_state.topic_information['pmids']
    article_data_df = st.session_state.topic_information['article_data_df']
    df = st.session_state.topic_information['df']

screened = st.session_state.topic_information.get('screening_results', df)[st.session_state.topic_information.get('screening_results', df)['human_decision'] != 'Unscreened']
if len(screened) == 0:
    print('nothing screened?')
else:
    print(screened[:3])

if 'counts' in st.session_state.topic_information:
    screening_status = st.session_state.topic_information['counts']
else:
    screening_status = Counter(st.session_state.topic_information.get('screening_results', df)['human_decision'])
st.markdown(f'Fetched {count} documents. {screening_status["Unscreened"]} Unscreened, {screening_status["Include"]} Include, and {screening_status["Exclude"]} Exclude decisions (may lag behind fast screening)')

print(f'df columns {df.columns}')
keep_columns = ['pmid', 'human_decision', 'robot_ranking', 'title', 'abstract']
df = df[keep_columns]
edit_columns = ['human_decision']
frozen_columns = set(keep_columns) - set(edit_columns)

abstracts_only = st.checkbox('Only articles with abstracts')
screening_choice = st.radio('Show:', options=['All', 'Unscreened', 'Included', 'Excluded', 'Any processed'])
old_choices = dict(zip(df['pmid'], df['human_decision']))
match screening_choice:
    case 'Unscreened':
        display_df = df[df['human_decision'] == 'Unscreened']
    case 'Included':
        display_df = df[df['human_decision'] == 'Include']
    case 'Excluded':
        display_df = df[df['human_decision'] == 'Exclude']
    case 'Any processed':
        display_df = df[df['human_decision'] != 'Unscreened']
    case 'All' | _:
        display_df = df

if abstracts_only:
    display_df = display_df[~display_df['abstract'].isna()]
    display_df = display_df[display_df['abstract'].apply(lambda x: len(x) > 0)]

st.session_state.topic_information['current_screening']['pmids'] = list(display_df['pmid'])

screening_results = st.data_editor(
    display_df,
    column_config={
        'pmid': st.column_config.TextColumn(
            'PMID',
            help='pubmed ID',
            width='small',
        ),
        'human_decision': st.column_config.SelectboxColumn(
            'Screening',
            help='Screen in or out this result',
            width='small',
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
    },
    # only allow editing the screening decision
    disabled=frozen_columns,
    hide_index=True,
    num_rows='dynamic',
    use_container_width=True,
)

st.session_state.topic_information['screening_results'] = screening_results
pmids_pairs = zip(screening_results['pmid'], screening_results['human_decision'])
# don't bother to insert Unscreened results
st.session_state.topic_information['counts'] = Counter(screening_results['human_decision'])
pmids_pairs = list(filter(lambda x: x[1] != 'Unscreened', pmids_pairs))
if len(pmids_pairs) > 0:
    pmid, decision = zip(*pmids_pairs)
    print('saving screening results', Counter(decision))
    database_utils.insert_topic_human_screening_pubmed_results(
        st.session_state.topic_information['topic_uid'],
        dict(pmids_pairs),
    )
    df_ = st.session_state.topic_information['df']
    df_[df['pmid'] == pmid]['human_decision'] = decision
    st.session_state.topic_information['screening_results'] = screening_results

    # so we don't screen anything twice
    if 'current_screening' in st.session_state.topic_information and 'screened' in st.session_state.topic_information['current_screening']:
        del st.session_state.topic_information['current_screening']['screened']


if st.button("View Evidence Map"):
    print('saving screening results post submit button', Counter(st.session_state.topic_information['screening_results']['human_decision']))
    database_utils.insert_topic_human_screening_pubmed_results(st.session_state.topic_information['topic_uid'], dict(zip(st.session_state.topic_information['screening_results']['pmid'], st.session_state.topic_information['screening_results']['human_decision'])))
    st.switch_page('pages/6-evidence_map.py')

if st.button('Individual Article Screening'):
    st.switch_page('pages/5-individual_screening.py')

# TODO add a linkout to the pubmed article
# TODO reorder columns for better user experience
