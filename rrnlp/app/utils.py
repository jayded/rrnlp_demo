import streamlit as st
import rrnlp.app.database_utils as database_utils
import rrnlp.models.SearchBot as SearchBot
import rrnlp.models.ScreenerBot as ScreenerBot

st.title("Search Bot")

# same as get_trial_reader
@st.cache_resource
def get_searcher():
    search_bot = SearchBot.PubmedQueryGeneratorBot(device='cpu')
    return search_bot

@st.cache_resource
def get_screener():
    screener_bot = ScreenerBot.load_screener(device='cpu')
    return screener_bot

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
            st.session_state['topic_information']['topic_name'] = selected['topic_name']
            st.session_state['topic_information']['topic_uid'] = selected['topic_uid']
            st.session_state['topic_information']['search_prompt'] = selected['search_text']
            st.session_state['topic_information']['query'] = selected['search_query']
            st.session_state['topic_information']['finalize'] = selected.get('final', 0)

            
            if st.session_state['topic_information']['finalize'] == 1:
                st.switch_page('pages/4-search_results_and_screening.py')
            else:
                st.switch_page('pages/3-develop_search.py')
        elif len(selected_row) > 1:
            assert False, 'impossible selection of multiple topics! please stop trying to break the app, and pick one!'

        if st.button('Manage research topics', 'New research topics can be added here'):
            st.switch_page('pages/2-existing_projects.py')
    return current_topics
