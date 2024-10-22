import streamlit as st
import rrnlp.app.database_utils as database_utils
import rrnlp.models.SearchBot as SearchBot
import rrnlp.models.ScreenerBot as ScreenerBot
import rrnlp.models.NumericalExtractionBot as NumericalExtractionBot

st.title("Search Bot")

# same as get_trial_reader
@st.cache_resource
def get_searcher():
    return SearchBot.get_llama_cpp_bot()
    #search_bot = SearchBot.get_search_bot(
    #    #weights=st.config['search_bot']['model_path'],
    #    #tokenizer=st.config['search_bot']['tokenizer'],
    #    device='cpu',
    #)
    #return search_bot

@st.cache_resource
def get_screener():
    screener_bot = ScreenerBot.load_screener(device='cpu')
    return screener_bot

@st.cache_resource
def get_numerical_extractor_bot():
    numerical_extractor_bot = NumericalExtractionBot.get_numerical_extractor_bot()
    return numerical_extractor_bot

def extract_numerical_information(topic_uid):
    # TODO ideally run this and exclude previously extracted items
    all_included_pmids, evidence_map = database_utils.get_auto_evidence_map_from_topic_uid(st.session_state.topic_information['topic_uid'])
    evidence_map = evidence_map[['pmid', 'title', 'abstract', 'intervention', 'comparator', 'outcome']]
    evidence_map = evidence_map.rename(columns={'title': 'ti', 'abstract': 'ab'})

    extractor = get_numerical_extractor_bot()
    previous_extractions = database_utils.get_numerical_extractions_for_topic(topic_uid)
    previous_extracted_types = set([(row['pmid'], row['intervention'], row['comparator'], row['outcome']) for row in previous_extractions])

    extractions = []
    for row in evidence_map.to_dict(orient='records'):
        extraction_tuple = (row['pmid'], row['intervention'], row['comparator'], row['outcome'])
        if row in previous_extractions:
            continue
        ico_list = [{'intervention': row['intervention'], 'comparator': row['comparator'], 'outcome': row['outcome']}]
        extraction = extractor.predict_for_ab(row, ico_list)
        for extraction_ in extraction:
            extractions.append((
                row['pmid'],
                row['intervention'],
                row['comparator'],
                row['outcome'],
                extraction['outcome_type'],
                extraction.get('binary_result', None),
                extraction.get('continuous_result', None),
            ))
        previous_extractions.add(extraction_tuple)

    database_utils.insert_numerical_extractions(extractions)

    return database_utils.get_numerical_extractions_for_topic(topic_uid)



def projects_view():
    current_topics = database_utils.current_topics(st.session_state.uid)
    #print('current_topics', current_topics)
    if len(current_topics) > 0:
        st.markdown('Current topics, pick one or')
        topic_selection = st.dataframe(
            current_topics,
            # it would be nice to hide this
            #disabled=['topic_uid'],
            selection_mode='single-row',
            on_select='rerun',
            hide_index=True,
        )
        print('topic_selection', topic_selection)
        selected_row = topic_selection['selection']['rows']
        if len(selected_row) == 1:
            selected = current_topics[selected_row[0]]
            print(selected)
            if not hasattr(st.session_state, 'topic_information'):
                st.session_state['topic_information'] = {}
            # TODO use the database entry instead of the above
            for k, v in selected.items():
                st.session_state['topic_information'][k] = v
            #st.session_state['topic_information']['topic_name'] = selected['topic_name']
            #st.session_state['topic_information']['topic_uid'] = selected['topic_uid']
            #st.session_state['topic_information']['search_prompt'] = selected['search_text']
            #st.session_state['topic_information']['query'] = selected['search_query']
            #st.session_state['topic_information']['finalize'] = selected.get('final', 0)


            if st.session_state['topic_information']['final'] == 1:
                st.switch_page('pages/4-search_results_and_screening.py')
            else:
                st.switch_page('pages/3-develop_search.py')
        elif len(selected_row) > 1:
            assert False, 'impossible selection of multiple topics! please stop trying to break the app, and pick one!'

        if st.button('Manage research topics', 'New research topics can be added here'):
            st.switch_page('pages/2-existing_projects.py')
    return current_topics
