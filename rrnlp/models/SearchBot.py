import os
import sys 
import tempfile
import time
from typing import Dict, List, Tuple, Union

import numpy as np 

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

import sqlite3
#import pubmed_parser as pp
#from lxml import etree
#from pubmed_mapper import Article
from Bio import Entrez

import rrnlp
from rrnlp.models import get_device

#weights_path = rrnlp.models.weights_path
# TODO
#weights = '/media/more_data/jay/overflow/ei_demo/base_query_model/Review_title/checkpoint_epoch_0006'
weights = '/media/more_data/jay/overflow/ei_demo/RRnlp/ckpt6_merged'

#weights = '/media/more_data/jay/overflow/ei_demo/base_query_model/Review_title/checkpoint_epoch_0006'


def get_topic_to_pubmed_converter(weights, base_tokenizer, device) -> AutoModelForCausalLM:
    ''' 
    Returns the 'punchline' extractor, which seeks out sentences that seem to convey
    main findings. 
    '''
    device = get_device(device=device)
    #dtype = torch.float32
    #print('loading to', device)
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
    os.makedirs('./offload', exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2',
        #is_trainable=False,
        device_map='auto',
        offload_folder='./offload',
        low_cpu_mem_usage=True,
    )
    #model = base_model.load_adapter(
    #    weights,
    #    offload_folder=tempfile.TemporaryDirectory().name,
    #)
    #model = AutoPeftModelForCausalLM.from_pretrained(
    #    weights,
    #    device_map='auto',
    #    offload_folder=tempfile.TemporaryDirectory().name,
    #)
    #model = AutoPeftModelForCausalLM.from_pretrained(
    #    weights,
    #    is_trainable=False,
    #    #torch_dtype=dtype,
    #    #device=device,
    #    device_map='auto',
    #    #device_map='cpu',
    #    #load_in_8bit=True,
    #    #use_ram_optimized_load=False,
    #).merge_and_unload()
    #model = model.to(device)
  
    return model, tokenizer


class PubmedQueryGeneratorBot:
    
    def __init__(
        self,
        weights=weights,
        base_tokenizer='mistralai/Mistral-7B-Instruct-v0.2',
        device='auto',
    ):
        self.query_generator, self.tokenizer = get_topic_to_pubmed_converter(
            weights=weights,
            base_tokenizer=base_tokenizer,
            device=device,
        )
        self.query_generator.eval()

    def generate_review_topic(self, review_topic: str) -> str:
        title_inputs = self.tokenizer.apply_chat_template(
            [
                {
                    'role': 'user',
                    'content': f'Translate the following into a boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Prefer shorter queries.\n{review_topic}',
                }
            ],
            device=self.query_generator.device,
            return_tensors='pt'
        )
        #import ipdb; ipdb.set_trace()
        titlequery = self.query_generator.generate(
            input_ids=title_inputs,
            min_length=5,
            no_repeat_ngram_size=5,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=2048,
        )
        titlequery = self.tokenizer.decode(titlequery.squeeze())
        titlequery = titlequery.split('[/INST]')[-1]
        titlequery = titlequery.replace('</s>', '')
        return titlequery

    @classmethod
    def rct_filter(cls) -> str:
        """Cochrane precision/sensitivity balanced RCT filter

        See https://work.cochrane.org/pubmed for more details. Direct pubmed link: 
            https://pubmed.ncbi.nlm.nih.gov/?cmd=search&term=(randomized+controlled+trial%5Bpt%5D+OR+controlled+clinical+trial%5Bpt%5D+OR+randomized%5Btiab%5D+OR+placebo%5Btiab%5D+OR+clinical+trials+as+topic%5Bmesh%3Anoexp%5D+OR+randomly%5Btiab%5D+OR+trial%5Bti%5D+NOT+(animals%5Bmh%5D+NOT+humans+%5Bmh%5D))
        """
        return '(randomized controlled trial[pt] OR controlled clinical trial[pt] OR randomized[tiab] OR placebo[tiab] OR clinical trials as topic[mesh:noexp] OR randomly[tiab] OR trial[ti] NOT (animals[mh] NOT humans [mh]))'

    @classmethod
    def add_rct_filter(cls, query: str) -> str:
        """Add Cochrane precision/sensitivity balanced RCT filter

        See https://work.cochrane.org/pubmed for more details. Direct pubmed link: 
            https://pubmed.ncbi.nlm.nih.gov/?cmd=search&term=(randomized+controlled+trial%5Bpt%5D+OR+controlled+clinical+trial%5Bpt%5D+OR+randomized%5Btiab%5D+OR+placebo%5Btiab%5D+OR+clinical+trials+as+topic%5Bmesh%3Anoexp%5D+OR+randomly%5Btiab%5D+OR+trial%5Bti%5D+NOT+(animals%5Bmh%5D+NOT+humans+%5Bmh%5D))
        """
        return query + ' AND ' + cls.rct_filter()

    @classmethod
    def _clean_pmids(cls, pmids: Union[int, str, List[int], List[str]]):
        if isinstance(pmids, (str, int)):
            pmids = [str(pmids)]
            if pmids[0].count(',') > 10000:
                raise Exception('Entrez apis refuse to return more than 10k records, and refuse to allow starting retrieval after 10k')
        elif isinstance(pmids, list):
            if len(pmids) > 10000:
                raise Exception('Entrez apis refuse to return more than 10k records, and refuse to allow starting retrieval after 10k')
            pmids = list(map(str, pmids))
        else:
            assert False
        return pmids

    @classmethod
    def execute_pubmed_search(
            cls,
            query,
            email='deyoung.j@northeastern.edu',
            api_key=None,
            retmax=1000,
        ) -> Tuple[int, List[int]]:
        """Query pubmed and find PMIDs (count, list)
        """
        # TODO return the corrected query as well?
        # TODO cache
        if email is not None:
            Entrez.email = email
        if api_key is not None:
            Entrez.api_key = api_key
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(handle)
        count = record['Count']
        pmids = record['IdList']
        pmids = PubmedQueryGeneratorBot._clean_pmids(pmids)
        handle.close()
        return count, pmids

    #@classmethod
    #def fetch_pmcids(cls, pmids: Union[int, str, List[int], List[str]], email='deyoung.j@northeastern.edu') -> Tuple[Dict[str,str], Dict[str, str]]:
    #    """Get a mapping of pmid -> pmcids and visa versa"""
    #    if email is not None:
    #        Entrez.email = email
    #    pmids = PubmedQueryGeneratorBot._clean_pmids(pmids)
    #    pmids = ','.join(pmids)
    #    handle = Entrez.esummary(db="pubmed", id=pmids)
    #    records = Entrez.read(handle)
    #    pmid_to_pmcid = dict()
    #    pmcid_to_pmid = dict()
    #    for r in records:
    #        ids = r['ArticleIds']
    #        pmid_to_pmcid[ids['pubmed'][0]] = ids['pmc']
    #        pmcid_to_pmid[ids['pmc']] = ids['pubmed'][0]
    #    return pmid_to_pmcid, pmcid_to_pmid
        
    #@classmethod
    #def fetch_article_data(
    #    cls,
    #    pmids: Union[int, str, List[int], List[str]],
    #    pubmed_archive_db_path: str,
    #    email='deyoung.j@northeastern.edu',
    #):
    #    conn = sqlite3.connect(
    #        f'file:{pubmed_archive_db_path}?mode=ro',
    #        uri=True,
    #    #res = dict()
    #    ## TODO allow parsing a saved nxml (see https://pypi.org/project/pubmed-mapper/)
    #    ## TODO migrate to pubmed_parser
    #    #print(f'fetching {len(pmids)} pmids')
    #    #for pmid in pmids:
    #    #    if pmid is None or len(pmid.strip()) == 0:
    #    #        print('found no length pmid')
    #    #        continue
    #    #    print(f'fetching {pmid}')
    #    #    try:
    #    #        article = Article.parse_pmid(pmid)
    #    #        res[article.pmid] = dict(article.to_dict())
    #    #    except Exception as e:
    #    #        print('skipped', pmid)
    #    #        print(e)
    #    #    time.sleep(0.05)
    #    #return res
    #    

    #def fetch_nxml(pmids: Union[int, str, List[int], List[str]), email=None):
    #    """Fetch, cache articles, return map of pmid -> p
    #    """
    #    if email is not None:
    #        Entrez.email = email
    #    # TODO cache (or read from pubmed archive?)
    #    # TODO read from cache?
    #    # TODO return type?
    #    # fetch articles, save, return nxml outputs
    #    pmids = PubmedQueryGeneratorBot._clean_pmids(pmids)
    #    pmids = ','.join(pmids)
    #    handle = Entrez.efetch(db='pubmed', id=pmids)
    #    for pmid in pmids:
    #        # Retrieve PMCID

    #        # Download NXML
    #        nxml_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/nxml/"
    #        nxml_response = requests.get(nxml_url)
    #        
    #        # Save the NXML content to a file
    #        with open(f"{pmid}.nxml", 'wb') as f:
    #            f.write(nxml_response.content)
    #    assert False


    #def fetch_summary(self, pmids: Union[int, str, List[int], List[str])):
    #    if email is not None:
    #        Entrez.email = email
    #    # entrez esummary
    #    pass
