import argparse
import json
import math
import os
import sqlite3
import time

import pandas as pd
import tqdm

from collections import Counter
from datetime import datetime, timedelta
from Bio import Entrez

Entrez.email = "deyoung.j@northeastern.edu"
with open('/home/deyoung.j/ncbi_api_key', 'r') as inf:
    Entrez.api_key = inf.read().strip()

def fetch_info(pmids):
    # Fetch the article data in XML format
    try:
        handle = Entrez.efetch(db="pubmed", id=','.join(pmids), rettype="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        print('failure for', pmids)
        print(e)
        #import ipdb; ipdb.set_trace()
        return []
    ret = []
    for record in records['PubmedArticle']:
        pmid = str(record['MedlineCitation']['PMID'])
        article = record['MedlineCitation']['Article']
        years = [int(x['Year']) for x in article['ArticleDate']]
        if len(years) == 0:
            years = [int(x['Year']) for x in record['PubmedData']['History']]
        if len(years) > 0:
            year = str(min(years))
        else:
            year = record['MedlineCitation']['DateRevised']['Year']
        title = article['ArticleTitle']
        abstract = article['Abstract']['AbstractText'][0] if 'Abstract' in article else None
        if 'Year' in article['Journal']['JournalIssue']['PubDate']:
            pubdate = article['Journal']['JournalIssue']['PubDate']['Year'] 
            if 'Month' in article['Journal']['JournalIssue']['PubDate']:
                pubdate += '-' + article['Journal']['JournalIssue']['PubDate']['Month']
            if 'Day' in article['Journal']['JournalIssue']['PubDate']:
                pubdate += '-' + article['Journal']['JournalIssue']['PubDate']['Day']
        elif 'MedlineDate' in article['Journal']['JournalIssue']['PubDate']:
            pubdate = article['Journal']['JournalIssue']['PubDate']['MedlineDate'] 
        else:
            assert False

        d = {
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'year': year,
            'pubdate': pubdate,
            'publication_types': article['PublicationTypeList'],
            #'mesh_terms':
            #'keywords':
        }
        if 'ISOAbbreviation' in article['Journal']:
            d['journal'] = article['Journal']['ISOAbbreviation']
        elif 'Title' in article['Journal']:
            d['journal'] = article['Journal']['Title']
        else:
            import ipdb; ipdb.set_trace()
        ret.append(d)

    for record in records['PubmedBookArticle']:
        article = record['BookDocument']
        import ipdb; ipdb.set_trace()
        title = '' + article['Book']['BookTitle']
        abstract = '' + article['Abstract']['AbstractText'][0] if 'Abstract' in article else None
        year = article['Book']['PubDate']['Year']
        if 'Year' in article['Journal']['JournalIssue']['PubDate']:
            pubdate = article['Journal']['JournalIssue']['PubDate']['Year'] 
            if 'Month' in article['Journal']['JournalIssue']['PubDate']:
                pubdate += '-' + article['Journal']['JournalIssue']['PubDate']['Month']
            if 'Day' in article['Journal']['JournalIssue']['PubDate']:
                pubdate += '-' + article['Journal']['JournalIssue']['PubDate']['Day']
        elif 'MedlineDate' in article['Journal']['JournalIssue']['PubDate']:
            pubdate = article['Journal']['JournalIssue']['PubDate']['MedlineDate'] 
        else:
            assert False
        d = {
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'year': year,
            'pubdate': pubdate,
            'publication_types': article['PublicationTypeList'],
            #'mesh_terms':
            #'keywords':
        }
        if 'ISOAbbreviation' in article['Journal']:
            d['journal'] = article['Journal']['ISOAbbreviation']
        elif 'Title' in article['Journal']:
            d['journal'] = article['Journal']['Title']
        else:
            import ipdb; ipdb.set_trace()
        ret.append(d)

        ## Extract the title and abstract
        #if len(records['PubmedArticle']) > 0:
        #    article = records['PubmedArticle'][0]['MedlineCitation']['Article']
        #    #year = records['PubmedArticle'][0]['MedlineCitation']['DateCompleted']['Year']
        #    years = [int(x['Year']) for x in article['ArticleDate']]
        #    if len(years) == 0:
        #        years = [int(x['Year']) for x in records['PubmedArticle'][0]['PubmedData']['History']]
        #    if len(years) > 0:
        #        year = str(min(years))
        #    else:
        #        year = records['PubmedArticle'][0]['MedlineCitation']['DateRevised']['Year']
        #    title = article['ArticleTitle']
        #    abstract = article['Abstract']['AbstractText'][0] if 'Abstract' in article else None
        #elif len(records['PubmedBookArticle']) > 0:
        #    article = records['PubmedBookArticle'][0]['BookDocument']
        #    title = '' + article['Book']['BookTitle']
        #    abstract = '' + article['Abstract']['AbstractText'][0] if 'Abstract' in article else None
        #    year = article['Book']['PubDate']['Year']
        #elif len(records['PubmedBookArticle']) == 0 and len(records['PubmedArticle']) == 0:
        #    print('no records for:')
        #    print(pmid)
        #    print(records.keys())
        #    print(records)
        #    return None, None, None
        #else:
        #    # this state shouldn't be possible anymore
        #    print(pmid)
        #    print(records.keys())
        #    print(records)
        #    assert False
        #
        #if 'Year' in article['Journal']['JournalIssue']['PubDate']:
        #    pubdate = article['Journal']['JournalIssue']['PubDate']['Year'] 
        #    if 'Month' in article['Journal']['JournalIssue']['PubDate']:
        #        pubdate += '-' + article['Journal']['JournalIssue']['PubDate']['Month']
        #    if 'Day' in article['Journal']['JournalIssue']['PubDate']:
        #        pubdate += '-' + article['Journal']['JournalIssue']['PubDate']['Day']
        #elif 'MedlineDate' in article['Journal']['JournalIssue']['PubDate']:
        #    pubdate = article['Journal']['JournalIssue']['PubDate']['MedlineDate'] 
        #else:
        #    assert False

        #d = {
        #    'pmid': pmid,
        #    'title': title,
        #    'abstract': abstract,
        #    'year': year,
        #    'journal': article['Journal']['ISOAbbreviation'],
        #    'pubdate': pubdate,
        #    'publication_types': article['PublicationTypeList'],
        #    #'mesh_terms':
        #    #'keywords':
        #}
        #ret.append(d)
    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True, help='SQLite database resulting from processing pubmed nxml documents')
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--ids_to_process_json', type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, 'r') as inf:
            already_processed_pmids = set(
                map(
                    lambda x: x['pmid'],
                    map(json.loads,
                        filter(
                            lambda y: len(y.strip()) > 0,
                            inf
                        )
                    )
                )
            )
    else:
        already_processed_pmids = set()

    if os.path.exists(args.ids_to_process_json):
        with open(args.ids_to_process_json, 'r') as inf:
            pmids_missing_abstracts = set(json.loads(inf.read()))
    else:
        def namedtuple_factory(cursor, row):
            return dict(zip([x[0] for x in cursor.description], row))

        con = sqlite3.connect(
            args.database,
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        con.row_factory = namedtuple_factory
        articles_and_abstracts = con.execute('''
            select pmid, title, abstract from article_data
        ''').fetchall()

        pmids_missing_abstracts = set()
        for row in tqdm.tqdm(articles_and_abstracts):
            abstract = row['abstract']
            if abstract is None or abstract != abstract or len(abstract.strip()) == 0:
                pmids_missing_abstracts.add(row['pmid'])

        with open(args.ids_to_process_json, 'w') as of:
            of.write(json.dumps(list(pmids_missing_abstracts)))

    print(f'Found {len(pmids_missing_abstracts)} pmids, have already processed {len(already_processed_pmids)}')

    # n.b. this number starts off high, e.g. 5000, then gets reduced as this is more complete. This is to handle some items failing to be read.
    fetch_count=1
    with open(args.output_jsonl, 'a') as of:
        pmids_missing_abstracts = list(pmids_missing_abstracts - already_processed_pmids)
        for i in tqdm.tqdm(range(0, len(pmids_missing_abstracts), fetch_count)):
            results = fetch_info(pmids_missing_abstracts[i:i+fetch_count])
            for res in results: 
                of.write(json.dumps(res) + '\n')
        #for pmid in tqdm.tqdm(pmids_missing_abstracts):
        #    if pmid not in already_processed_pmids:
        #        res = fetch_info(pmid)
        #        of.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
