import logging
import os
import sys
import threading
import time

from os.path import abspath, dirname
from typing import List

import yaml

from celery import Celery
from yaml.loader import SafeLoader

#from rrnlp.app.utils import get_searcher
import rrnlp.models.SearchBot as SearchBot
import rrnlp.models.MultiSummaryGenerationBot as MultiSummaryGenerationBot


config_file = '/data/ei_demo/RRnlp/rrnlp/app/demo_pw.yml'
#config_file = sys.argv[1]
with open(config_file, 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)
# TODO real config for this somehow
base_dir = dirname(dirname(dirname(abspath(__file__))))
sqlite_path = f'sqlalchemy+sqlite:////{base_dir}/broker.sqlite3'
result_backend = f'db+sqlite:////{base_dir}/backend.sqlite3'

app = Celery('rrnlp',
             broker=sqlite_path,
             backend=result_backend,
             #backend='rpc://',
             #include=['tasks'])
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        #logging.FileHandler('celery.log'),
        logging.StreamHandler()
    ]
)

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

logging.info(f'celery paths: broker: {sqlite_path}, backend: {result_backend}')
logging.info(f'Consider if sqlite is still appropriate for these paths')

_model_loaded = False

@app.on_after_configure.connect
def setup_text_generation(sender, **kwargs):
    global searcher, mds_summarizer, _model_loaded
    if _model_loaded:
        logging.info(f'skipping loading, model already loaded')
    else:
        _model_loaded = True
        get_searcher_start = time.time()
        #searcher = SearchBot.PubmedQueryGeneratorBot()
        searcher = SearchBot.get_topic_to_pubmed_converter(
            #weights=st.config['search_bot']['model_path'],
            #tokenizer=st.config['search_bot']['tokenizer'],
            #device='cpu',
        )
        get_searcher_end = time.time()
        logging.info(f'loading the searcher took {get_searcher_end - get_searcher_start} seconds')
        get_summarizer_start = time.time()
        logging.info(f'loading key from {config["openai_api_key"]}')
        mds_summarizer = MultiSummaryGenerationBot.load_openai_summary_bot(api_key_file = config['openai_api_key'])
        #mds_summarizer = MultiSummaryGenerationBot.load_mds_summary_bot()
        get_summarizer_end = time.time()
        logging.info(f'loading the mds summarizer took {get_summarizer_end - get_summarizer_start} seconds')
        _model_loaded = True

@app.task
def generate_search(search_text):
    generate_start = time.time()
    #with app.search_semaphore:
    query = searcher.generate_review_topic(search_text)
    generate_end = time.time()
    logging.info(f'generating the query took {generate_end - generate_start} seconds: {query}')
    return query
 
@app.task
def mds_summarize(search_text, input_texts: List[str]):
    generate_start = time.time()
    #with app.search_semaphore:
    summary, response = mds_summarizer.predict_for_docs(search_text, input_texts)
    generate_end = time.time()
    logging.info(f'generating the summary took {generate_end - generate_start} seconds')
    return summary
    
## Other RRNlp generations
#@app.task
#def process_pmid(pmid):
#    # TODO: all the things
#    pass

if __name__ == '__main__':
    #app.search_semaphore = threading.Semaphore(1)
    app.start(argv=sys.argv[1:])

