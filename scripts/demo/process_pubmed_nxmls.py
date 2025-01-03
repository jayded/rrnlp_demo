import argparse
import ast
import glob
import json
import os
import gzip

import xml.etree.cElementTree as ET
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict


import pandas as pd
import pubmed_parser as pp
import torch

from tqdm import tqdm

import rrnlp
from rrnlp import TrialReader

skip_pmids = {'35443369'}


def main():
    parser = argparse.ArgumentParser(description='decode flan over a csv of inputs')
    parser.add_argument('--input_gzs', nargs='+', required=True, help='compressed pubmed downloaded nxml to be read')
    parser.add_argument('--previous_output_jsonls', required=False, help='dir to find a jsonl output of this previous run')
    parser.add_argument('--output_csv', required=True, help='csv dirs')
    parser.add_argument('--updates', required=False, action='store_true', default=False, help='are we processing an updates file?')
    parser.add_argument('--additional_ids', required=False, help='newline separated additional ids for additional processing')
    #parser.add_argument('--additional_article_info', required=False, help='jsonl file specifying article abstracts & some other pubmed metadata (some articles do not have information in the pubmed dumps but do on the website, as of nov 2024 something like 11 million documents have this property)')
    args = parser.parse_args()

    if args.additional_ids is not None:
        with open(args.additional_ids, 'r') as inf:
            additional_ids = set(map(str.strip, inf.readlines()))
    else:
        additional_ids = set()
    
    #additional_article_data = dict()
    #if args.additional_article_info is not None:
    #    with open(args.additional_article_info, 'r') as inf:
    #        for line in inf:
    #            if len(line.strip()) == 0:
    #                continue
    #            contents = json.loads(line)
    #            additional_article_data[str(contents['pmid'])] = contents

    processed_ids = set()
    rows = []
    json_rows = []
    in_output_df = set()
    if os.path.exists(args.output_csv):
        if args.output_csv.endswith('parquet'):
            try:
                df = pd.read_parquet(args.output_csv)
            except:
                df = pd.read_parquet(args.output_csv + '.bak')
        else:
            df = pd.read_csv(args.output_csv)
        rows = df.to_dict(orient='records')
        processed_ids = set(df['pmid'])
        in_output_df.update(set(df['pmid']))
    json_out_file_path = args.output_csv + '.jsonl'
    json_out_file = open(json_out_file_path, 'a+')
    tasks = ('rct_bot', 'pico_span_bot', 'bias_ab_bot', 'ico_ev_bot', 'sample_size_bot', 'study_design_bot')
    # TODO enable numerical extraction
    #tasks = ('rct_bot', 'pico_span_bot', 'bias_ab_bot', 'ico_ev_bot', 'sample_size_bot', 'study_design_bot', 'numerical_extraction_bot')
    reader = TrialReader(tasks=tasks, device='dynamic')
    delete_ids = set()

    for gz_file in args.input_gzs:
        print(f'processing {gz_file}')
        previous_outputs = dict()
        if args.previous_output_jsonls is not None:
            previous_output_jsonls_f = os.path.join(args.previous_output_jsonls, os.path.basename(args.output_csv) + '.jsonl')
            print(f'Reading previous_outputs from {previous_output_jsonls_f}')
            with open(previous_output_jsonls_f, 'r') as inf:
                previous_outputs = map(json.loads, filter(lambda x: len(x) > 0, map(str.strip, inf.readlines())))
                previous_outputs = {x['pmid']:x for x in previous_outputs}
        else:
            previous_outputs = dict()
        # in case this job started/stopped in the middle, we can rescue the outputs and not rerun them.
        # ask me why this is important...
        if os.path.exists(json_out_file_path):
            print(f'Job appears to have restarted, reading from {json_out_file_path}')
            with open(json_out_file_path, 'r') as inf:
                for i, line in enumerate(inf):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    line = json.loads(line)
                    # rerun every one of these..
                    #if 'ico_ev_bot' in line:
                    #    del line['ico_ev_bot']
                    #if 'rct_bot' in line:
                    #    del line['rct_bot']
                    previous_outputs[line['pmid']] = line
                    processed_ids.add(line['pmid'])
            print(f'Have restored {len(previous_outputs)} outputs')
            processed_ids = set(previous_outputs.keys())
            #processed_ids = set()
        if args.updates:
            with open(gz_file, 'rb') as gz:
                decompressedFile = gzip.GzipFile(fileobj=gz, mode='r')
                for event, elem in ET.iterparse(decompressedFile, events=("start", "end")):
                    if elem.tag == "DeleteCitation" and event=="end":
                        for r in elem:
                            delete_ids.add(r.text)
        for i, article_info in tqdm(enumerate(pp.parse_medline_xml(gz_file))):
            title = article_info['title']
            if 'Erratum' in title:
                continue
            abstract = article_info['abstract']
            pmid = article_info['pmid']
            if article_info.get('delete', False):
                print(f'skipping {pmid}, says delete!')
                delete_ids.add(pmid)
                continue
            if pmid in delete_ids:
                print(f'skipping {pmid}, says delete!')
                continue
            if pmid in skip_pmids:
                print(f'skipping {pmid}, manual choice (processing issue)!')
                continue
            #if pmid in processed_ids:
            #    if pmid in in_output_df:
            #        continue
            previous_output = previous_outputs.get(pmid, None)
            #if previous_output is not None:
            #    print(f'Resurrecting input for {pmid}')
            #    #print(f'Resurrecting input for {pmid}: {previous_output}')
            mesh_terms = article_info['mesh_terms']
            keywords = article_info['keywords']
            publication_types = article_info['publication_types']
            journal = article_info['journal']
            pubdate = article_info['pubdate']
            if pmid in previous_outputs:
                res = previous_outputs[pmid]
                #if len(abstract) == 0 or len(res['abstract']) == 0 and pmid in additional_article_data:
                #    abstract = additional_article_data[pmid]['abstract']
                #    if abstract is not None and abstract == abstract and len(abstract.strip()) > 0:
                #        res = reader.read_trial(
                #            ab={
                #                'ti': title,
                #                'ab': abstract,
                #            },
                #            task_list=tasks,
                #            process_rcts_only=not (pmid in additional_ids),
                #            previous=previous_output,
                #        )
                #    else:
                #        res = previous_outputs[pmid]
                #else:
                #    res = previous_outputs[pmid]
            else:
                res = reader.read_trial(
                    ab={
                        'ti': title,
                        'ab': abstract,
                    },
                    task_list=tasks,
                    process_rcts_only=not (pmid in additional_ids),
                    previous=previous_output,
                )
            res['ti'] = title
            res['ab'] = abstract
            res['pmid'] = pmid
            res['mesh_terms'] = mesh_terms
            res['publication_types'] = publication_types
            res['keywords'] = keywords
            res['journal'] = journal
            res['pubdate'] = pubdate

            base = {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'mesh_terms': json.dumps(mesh_terms),
                'publication_types': json.dumps(publication_types),
                'keywords': json.dumps(keywords),
                'journal': journal,
                'pubdate': pubdate,
            }
            #tasks = ['rct_bot', 'pico_span_bot', 'bias_ab_bot', 'ico_ev_bot', 'sample_size_bot', 'study_design_bot']
            # rct_bot
            base['is_rct'] = int(res['rct_bot']['is_rct'])
            base['prob_rct'] = res['rct_bot']['prob_rct']
            for k, v in res['rct_bot']['scores'].items():
                base[k] = int(v)
            # bias_ab_bot
            if 'bias_ab_bot' in res:
                assert len(res['bias_ab_bot']) == 1
                base['prob_lob_rob'] = res['bias_ab_bot']['prob_low_rob']
            # sample_size_bot
            if 'sample_size_bot' in res:
                assert len(res['sample_size_bot']) == 1
                base['num_randomized'] = res['sample_size_bot']['num_randomized']

            if 'study_design_bot' in res:
                for k, v in res['study_design_bot'].items():
                    if isinstance(v, bool):
                        v = int(v)
                    if isinstance(v, (tuple, list, set, dict)):
                        assert False
                    base[k] = v

            # pico_span_bot
            # ['p', 'p_mesh', 'i', 'i_mesh', 'o', 'o_mesh']
            if 'pico_span_bot' in res:
                for typ in ['p', 'i', 'o']:
                    base[typ] = json.dumps(res['pico_span_bot'][typ])
                    base[f'{typ}_mesh'] = json.dumps(res['pico_span_bot'][f'{typ}_mesh'])

            if 'ico_ev_bot' in res:
                base['ico'] = json.dumps(res['ico_ev_bot'])
            rows.append(base)
            # no need to repeat an output in these files
            #if not (pmid in previous_outputs):
            if pmid not in previous_outputs:
                json_rows.append(json.dumps(res))
            processed_ids.add(pmid)
            if i % 100 == 0:
                df = pd.DataFrame(rows)
                if os.path.exists(args.output_csv):
                    os.rename(args.output_csv, args.output_csv + '.bak')
                if args.output_csv.endswith('parquet'):
                    df.to_parquet(args.output_csv)
                else:
                    df.to_csv(args.output_csv)
                json_out_file.write('\n'.join(json_rows) + '\n')
                json_rows = []
    df = pd.DataFrame(rows)
    os.rename(args.output_csv, args.output_csv + '.bak')
    if args.output_csv.endswith('parquet'):
        df.to_parquet(args.output_csv)
    else:
        df.to_csv(args.output_csv)
    json_out_file.write('\n'.join(json_rows) + '\n')
    json_out_file.close()
    os.remove(args.output_csv + '.bak')
    if len(delete_ids) > 0:
        with open(args.output_csv + '.delete_ids.json', 'w') as of:
            of.write(json.dumps(list(delete_ids)))


if __name__ == "__main__":
    main()

