import streamlit as st
import rrnlp.app.database_utils as database_utils
from rrnlp.app.celery_app import mds_summarize as celery_mds_summarize

if 'uid' not in st.session_state:
    st.switch_page('streamlit_app.py')
if 'topic_information' not in st.session_state or 'topic_uid' not in st.session_state.topic_information or 'topic_name' not in st.session_state.topic_information or 'final' not in st.session_state.topic_information:
    st.switch_page('pages/2-existing_projects.py')

if st.session_state.topic_information['final'] != 1:
    st.switch_page('pages/3-develop_search.py')

# mystery: why does the st module not...seem cooperate?
if not st.session_state.get('loaded_config', False):
    database_utils.load_config()

# display topic information
st.markdown(f'Automatically generated summary for {st.session_state.topic_information["topic_name"]}')

# TODO these dataframes should be specific to the summarization
count, pmids, article_data_df, df = database_utils.get_persisted_pubmed_search_and_screening_results(st.session_state.topic_information['topic_uid'])
st.session_state.topic_information['count'] = count
st.session_state.topic_information['pmids'] = pmids
st.session_state.topic_information['article_data_df'] = article_data_df
st.session_state.topic_information['df'] = df
# display included documents
df = df[df['human_decision'] != 'Unscreened']
pre_titles_len = len(df)
df = df[~df['title'].isna()]
post_titles_len = len(df)
df = df[~df['abstract'].isna()]
post_abstracts_len = len(df)
print(f'initial: {pre_titles_len}, post titles / pre abstracts: {post_titles_len}, abstracts: {post_abstracts_len}')

if len(df) == 0:
    st.markdown("Cannot generate a summary when no documents have been screened in")
    st.stop()

# TODO openai chatgpt summaries
# TODO column config & pretty & allow exclusions from the summaries
st.dataframe(df)
# call the summarizer?
df_ = df[df['human_decision'] == 'Include']
title_and_article = [title + '\n' + article for title, article in zip(df_['title'], df_['abstract'])]
if sum(map(len,map(str.split, title_and_article))) > 128000:
    st.markdown(f'Input texts are too long to generate a summary!')
else:
    # TODO more info in count
    with st.spinner(f'Generating a summary from {len(df_)} articles (note: {pre_titles_len - len(df)} articles removed for lacking a title/abstract)'):
        st.markdown("Potential summary (this is neither verified nor guaranteed to be accurate)")
        generated_summary = celery_mds_summarize(st.session_state.topic_information['search_text'], title_and_article)
        st.markdown(generated_summary)
