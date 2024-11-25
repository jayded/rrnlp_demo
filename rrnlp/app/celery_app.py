import logging
import os
import sys
import threading
import time

from os.path import abspath, dirname
from typing import List

import celery
import yaml

from celery import Celery, shared_task, Task
from yaml.loader import SafeLoader

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
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


@celery.signals.after_setup_logger.connect
def on_after_setup_logger(**kwargs):
    logger = logging.getLogger('celery')
    logger.propagate = True
    logger = logging.getLogger('celery.app.trace')
    logger.propagate = True

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

logging.info(f'celery paths: broker: {sqlite_path}, backend: {result_backend}')
logging.info(f'Consider if sqlite is still appropriate for these paths')


class SearchTask(Task):
    def __init__(self):
        print('loading searcher')
        get_searcher_start = time.time()
        self.searcher = SearchBot.get_topic_to_pubmed_converter()
        get_searcher_end = time.time()
        logging.info(f'loading the searcher took {get_searcher_end - get_searcher_start} seconds')

    def compute(self, search_text):
        generate_start = time.time()
        logging.info(f'model data: {id(self.searcher)}')
        logging.info(f'search_text: {search_text}')
        query = self.searcher.generate_review_topic(search_text)
        generate_end = time.time()
        logging.info(f'generating the query took {generate_end - generate_start} seconds: {query}')
        return query

@app.task(base=SearchTask, name='generate_search', acks_late=False, bind=True)
def generate_search(self: Task, search_text: str):
    return self.compute(search_text)

class MDSSummarizer(Task):
    def __init__(self):
        get_summarizer_start = time.time()
        logging.info(f'loading key from {config["openai_api_key"]}')
        self.mds_summarizer = MultiSummaryGenerationBot.load_openai_summary_bot(api_key_file = config['openai_api_key'])
        get_summarizer_end = time.time()
        logging.info(f'loading the mds summarizer took {get_summarizer_end - get_summarizer_start} seconds')

    def compute(self, search_text: str, input_texts: List[str]):
        generate_start = time.time()
        summary, response = self.mds_summarizer.predict_for_docs(search_text, input_texts)
        generate_end = time.time()
        logging.info(f'generating the summary took {generate_end - generate_start} seconds')
        return summary

@app.task(base=MDSSummarizer, name='mds_summarize', acks_late=False, bind=True)
def mds_summarize(self: Task, search_text, input_texts: List[str]):
    return self.compute(search_text, input_texts)

if __name__ == '__main__':
    app.start(argv=sys.argv[1:])

