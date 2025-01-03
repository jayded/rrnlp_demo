import argparse
import ast
import copy
import glob
import json
import os
import time

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import sqlite3
import tqdm


def main():
    parser = argparse.ArgumentParser(description='take the output of process_pubmed_nxmls.py and add it to an sqlite database')
    parser.add_argument('--input_parquets', nargs='+', required=True, help='parquet extraction outputs')
    parser.add_argument('--input_jsonls', nargs='+', required=True, help='matching jsonl outputs')
    parser.add_argument('--output_database', required=True, help='output to an sqlite file')
    parser.add_argument('--delete_ids', required=False, nargs='+', help='jsons of update file deletions')
    args = parser.parse_args()

    all_delete_ids = set()
    for x in args.delete_ids:
        if os.path.exists(x):
            with open(x, 'r') as inf:
                all_delete_ids.update(json.loads(inf.read()))

    assert len(args.input_parquets) == len(args.input_jsonls)

    con = sqlite3.connect(args.output_database)
    con.execute('pragma journal_mode = WAL;')
    con.execute('pragma synchronous = normal;')

    start = time.time()
    for df_file, jsonl_file in zip(args.input_parquets, args.input_jsonls):
        df = pd.read_parquet(df_file)
        article_data_rows = []
        #rct_bot_rows = []
        ico_ev_rows = []
        pico_rows = []
        study_design_bot_rows = []
        mesh_terms_rows = []
        publication_types_terms_rows = []
        #misc_extractions_rows = []
        all_pmids = set()
        total_failed_productions = {}
        # parse this file first so we have the most recent version (some articles might be written twice due to ...fun... processing situations
        lines = {}
        start_read_jsonl = time.time()
        with open(jsonl_file, 'r') as inf:
            for line in inf:
                line = line.strip()
                if len(line) == 0:
                    continue
                contents = json.loads(line)
                pmid = contents['pmid']
                lines[pmid] = contents
        end_read_jsonl = time.time()
        print(f'reading jsonl took {end_read_jsonl - start_read_jsonl} seconds')
        start_process_jsonl = time.time()
        for pmid, contents in tqdm.tqdm(lines.items()):
                if pmid in all_pmids:
                    print(f'Skipping entry for {pmid}, already processed')
                    continue
                all_pmids.add(pmid)
                journal_info_row = df[df['pmid'] == pmid].to_dict('records')[-1]
                #rct_bot_scores = {'pmid': pmid, 'is_rct': int(contents['rct_bot']['is_rct'])}
                #rct_bot_scores.update({k:int(v) for k,v in contents['rct_bot']['scores'].items()})
                #rct_bot_rows.append(rct_bot_scores)
                if 'ico_ev_bot' in contents:
                    for ico_ev_extraction in contents['ico_ev_bot']:
                        # attempt to handle misparsed/outputs with single quotes
                        if 'production' in ico_ev_extraction:
                            production = ico_ev_extraction['production']
                            production = production.replace('"', '\\"')
                            production = production.replace("['", '["')
                            production = production.replace("']", '"]')
                            production = production.replace("', '", '", "')
                            production = production.replace("','", '","')
                            try:
                                decoded = ast.literal_eval(production)
                                if len(decoded) == 0:
                                    print(f'No ICO/Evidence relations found for {production}')
                                for res in decoded:
                                    if len(res) != 5:
                                        print(f'potential error parsing {production}')
                                        continue
                                    try:
                                        label = res[-1].split('[LABEL]')[1].split('[OUT]')[0].strip()
                                    except:
                                        print(f'label parse error for {production}')
                                        label = None
                                    components = ["intervention", "outcome", "comparator", "evidence", "sentence", 'label']
                                    ico_tuplet = dict(zip(components, res))
                                    ico_tuplet['label'] = label
                                    ico_tuplet['pmid'] = pmid
                                    del ico_tuplet['sentence']
                                    ico_ev_rows.append(ico_tuplet)
                            except Exception as e:
                                print(f'failed to decode {production}')
                                total_failed_productions[pmid] = production
                                #import ipdb; ipdb.set_trace()
                        else:
                            ico_ev_extraction['pmid'] = pmid
                            ico_ev_rows.append(copy.deepcopy(ico_ev_extraction))

                    del contents['ico_ev_bot']
                if 'pico_span_bot' in contents:
                    for typ in ['p', 'i', 'o']:
                        for value in contents['pico_span_bot'].get(typ, []):
                            pico_rows.append({
                                'pmid': pmid,
                                'type': typ,
                                'value': value,
                            })
                        for value in contents['pico_span_bot'].get(f'{typ}_mesh', []):
                            v = {
                                'pmid': pmid,
                                'type': f'{typ}_mesh',
                            }
                            v.update(value)
                            pico_rows.append(v)
                    del contents['pico_span_bot']
                if 'study_design_bot' in contents:
                    contents['study_design_bot']['pmid'] = pmid
                    study_design = {'pmid': pmid}
                    study_design.update(contents['study_design_bot'])
                    for key in study_design.keys():
                        if isinstance(study_design[key], bool):
                            study_design[key] = int(study_design[key])
                    # duplicate the rct_bot information
                    for k, v in contents['rct_bot'].items():
                        if isinstance(v, dict):
                            continue
                        elif isinstance (v, bool):
                            study_design['rct_bot_' + k] = int(v)
                        else:
                            study_design['rct_bot_' + k] = v
                            
                    for k, v in contents['rct_bot']['scores'].items():
                        if isinstance(v, bool):
                            study_design['rct_bot_' + k] = int(v)
                        else:
                            study_design['rct_bot_' + k] = v
                    # wong place, but whatever
                    for extraction_type in ['bias_ab_bot', 'sample_size_bot']:
                        if extraction_type in contents:
                            for k, v in contents[extraction_type].items():
                                if v is not None:
                                    study_design[k] = v
                        study_design_bot_rows.append(study_design)
                        del contents[extraction_type]
                    del contents['study_design_bot']
                ## misc extractions
                #misc_extractions = {'pmid': pmid}
                ## wrong place, but good enough
                #for extraction_type in ['bias_ab_bot', 'sample_size_bot']:
                #    if extraction_type in contents:
                #        for k, v in contents[extraction_type].items():
                #            if v is not None:
                #                misc_extractions[k] = v
                #    del contents[extraction_type]
                #if len(misc_extractions) > 0:
                #    misc_extractions_rows.append(misc_extractions)
                article_data = {
                    'pmid': pmid,
                    'title': contents['ti'],
                    'abstract': contents['ab'],
                    'is_rct': int(contents['rct_bot']['is_rct']),
                    'prob_rct': contents['rct_bot']['prob_rct'],
                }
                for k, v in contents['rct_bot']['scores'].items():
                    article_data[k] = int(v)
                for entry in ['journal', 'pubdate', 'publication_types', 'mesh_terms', 'keywords']:
                    value = journal_info_row[entry]
                    value = value.strip()
                    if len(value) > 0:
                        article_data[entry] = value
                article_data_rows.append(article_data)
                # these terms come from elsewhere or should have but original rounds of processing did not include this information.
                if 'mesh_terms' in contents:
                    try:
                        mesh_terms = json.loads(contents['mesh_terms'])
                    except:
                        mesh_terms = json.loads(article_data['mesh_terms'])
                else:
                    mesh_terms = json.loads(article_data['mesh_terms'])
                if isinstance(mesh_terms, str):
                    mesh_terms = list(map(str.strip, mesh_terms.split(';')))
                for mesh_term in mesh_terms:
                    mesh_terms_rows.append({'pmid': pmid, 'mesh': mesh_term})

                if 'publication_types' in contents:
                    try:
                        publication_types = json.loads(contents['publication_types'])
                    except:
                        publication_types = json.loads(article_data['publication_types'])
                else:
                    publication_types = json.loads(article_data['publication_types'])
                if isinstance(publication_types, str):
                    publication_types = list(map(str.strip, publication_types.split(';')))
                for term in publication_types:
                    publication_types_terms_rows.append({'pmid': pmid, 'mesh': term})

                for field in ['ti', 'ab', 'rct_bot', 'mesh_terms', 'publication_types', 'keywords', 'journal', 'pubdate']:
                    if field in contents:
                        del contents[field]
                # leaving just the pmid
                assert len(contents) == 1, contents
        finish_process_jsonl = time.time()
        print(f'procesing jsonl took {finish_process_jsonl - start_process_jsonl} seconds')
        tables = [x[0] for x in con.execute("select name from sqlite_master where type='table'").fetchall()]
        print(f'failed parsing {len(total_failed_productions)}')
        print(f'have tables: {tables}')
        all_pmids.update(all_delete_ids)
        start_db = time.time()
        article_data_rows_df = pd.DataFrame(article_data_rows)
        if 'article_data' in tables:
            start_delete = time.time()
            con.execute(f'delete from article_data where pmid in ({",".join(all_pmids)})')
            end_delete = time.time()
            print(f'deleting article data took {end_delete - start_delete} seconds')
        if len(article_data_rows_df) > 0:
            print(f'Extracted {len(article_data_rows_df)} article data')
            article_data_rows_df.to_sql(
                name='article_data',
                con=con,
                if_exists='append',
                index=False,
                #index=True,
                #index_label='pmid',
                chunksize=30000,
            )

        end_db = time.time()
        print(f'writing article data took {end_db - start_db} seconds')
        #con.execute(f'delete from rct_bot where pmid in ({",".join(all_pmids)})')
        #rct_bot_rows_df = pd.DataFrame(rct_bot_rows)
        #if len(rct_bot_rows_df) > 0:
        #    rct_bot_rows_df.to_sql(
        #        name='rct_bot',
        #        con=con,
        #        if_exists='append',
        #        index=True,
        #        index_label='pmid',
        #        chunksize=1000,
        #    )
        #    con.commit()
        start_db = time.time()
        if 'ico_ev_bot' in tables:
            start_delete = time.time()
            con.execute(f'delete from ico_ev_bot where pmid in ({",".join(all_pmids)})')
            end_delete = time.time()
            print(f'deleting article data took {end_delete - start_delete} seconds')
        ico_ev_rows_df = pd.DataFrame(ico_ev_rows)
        if len(ico_ev_rows_df) > 0:
            print(f'Extracted {len(ico_ev_rows_df)} ico tuplets')
            ico_ev_rows_df.to_sql(
                name='ico_ev_bot',
                con=con,
                if_exists='append',
                index=False,
                #index=True,
                #index_label='pmid',
                chunksize=30000,
            )
        end_db = time.time()
        print(f'writing ico ev data took {end_db - start_db} seconds')
        start_db = time.time()
        if 'pico_bot' in tables:
            start_delete = time.time()
            con.execute(f'delete from pico_bot where pmid in ({",".join(all_pmids)})')
            end_delete = time.time()
            print(f'deleting article data took {end_delete - start_delete} seconds')
        pico_rows_df = pd.DataFrame(pico_rows)
        if len(pico_rows_df) > 0:
            print(f'Extracted {len(pico_rows_df)} pico rows')
            pico_rows_df.to_sql(
                name='pico_bot',
                con=con,
                if_exists='append',
                index=False,
                #index=True,
                #index_label='pmid',
                chunksize=30000,
            )
        end_db = time.time()
        print(f'writing pico data took {end_db - start_db} seconds')
        start_db = time.time()
        if 'study_design_bot' in tables:
            start_delete = time.time()
            con.execute(f'delete from study_design_bot where pmid in ({",".join(all_pmids)})')
            end_delete = time.time()
            print(f'deleting article data took {end_delete - start_delete} seconds')
        study_design_bot_rows_df = pd.DataFrame(study_design_bot_rows)
        if len(study_design_bot_rows) > 0:
            print(f'Extracted {len(study_design_bot_rows)} study design info')
            study_design_bot_rows_df.to_sql(
                name='study_design_bot',
                con=con,
                if_exists='append',
                index=False,
                #index=True,
                #index_label='pmid',
                chunksize=30000,
            )
        end_db = time.time()
        print(f'writing study design data took {end_db - start_db} seconds')
        start_db = time.time()
        if 'mesh_terms' in tables:
            start_delete = time.time()
            con.execute(f'delete from mesh_terms where pmid in ({",".join(all_pmids)})')
            end_delete = time.time()
            print(f'deleting article data took {end_delete - start_delete} seconds')
        mesh_terms_rows_df = pd.DataFrame(mesh_terms_rows)
        if len(mesh_terms_rows) > 0:
            print(f'Extracted {len(mesh_terms_rows)} mesh terms')
            mesh_terms_rows_df.to_sql(
                name='mesh_terms',
                con=con,
                if_exists='append',
                index=False,
                #index_label='pmid',
                chunksize=30000,
            )

        end_db = time.time()
        print(f'writing mesh terms took {end_db - start_db} seconds')
        start_db = time.time()
        if 'publication_types' in tables:
            start_delete = time.time()
            con.execute(f'delete from publication_types where pmid in ({",".join(all_pmids)})')
            end_delete = time.time()
            print(f'deleting article data took {end_delete - start_delete} seconds')
        publication_types_terms_rows_df = pd.DataFrame(publication_types_terms_rows)
        if len(publication_types_terms_rows) > 0:
            print(f'Extracted {len(mesh_terms_rows)} mesh terms')
            publication_types_terms_rows_df.to_sql(
                name='publication_terms',
                con=con,
                if_exists='append',
                index=False,
                #index=True,
                #index_label='pmid',
                chunksize=30000,
            )
        end_db = time.time()
        print(f'writing publication types took {end_db - start_db} seconds')
        start_db = time.time()
        con.commit()
        con.close()
        end_db = time.time()
        print(f'final commit {end_db - start_db} seconds')
    end = time.time()
    print(f'total processing time {end - start} seconds')
        
if __name__ == '__main__':
    main()
