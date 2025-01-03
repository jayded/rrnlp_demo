import argparse
import glob
import json
import os

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import sqlite3


def main():
    parser = argparse.ArgumentParser(description='take the output of process_pubmed_nxmls.py and add it to an sqlite database')
    parser.add_argument('--input_parquets', nargs='+', required=True, help='parquet extraction outputs')
    parser.add_argument('--output_database', required=True, help='output to an sqlite file')
    args = parser.parse_args()

    con = sqlite3.connect(args.output_database)

    # TODO create a second table for p,i,c,o elems?
    for df_path in args.input_parquets:
        print(f'processing {df_path}')
        all_df = pd.read_parquet(df_path)
        ico_re_rows = []
        for i, row in all_df.iterrows():
            if row['ico'] is None:
                continue
            ico_elems = json.loads(row['ico'])
            for e in ico_elems:
                e['pmid'] = row['pmid']
                ico_re_rows.append(e)
                    
        if 'ico' in all_df.columns:
            del all_df['ico']
        all_df.to_sql(
            name='pubmed_extractions',
            con=con,
            if_exists='append',
            #index=True,
            #index_label='pmid',
            chunksize=1000,
        )
        con.commit()
        icos_df = pd.DataFrame(ico_re_rows)
        if len(icos_df) > 0:
            icos_df.to_sql(
                name='pubmed_extractions_ico_re',
                con=con,
                if_exists='append',
                #index=True,
                #index_label='pmid',
                chunksize=1000,
            )
            con.commit()
    con.close()


if __name__ == "__main__":
    main()
