import argparse
import json

import warnings
warnings.filterwarnings("ignore")

import sqlite3

def main():
    parser = argparse.ArgumentParser(description='delete pmids from pubmed db')
    parser.add_argument('--delete_jsons', nargs='+', required=True, help='jsons to clear from the pubmed database')
    parser.add_argument('--output_database', required=True, help='output to an sqlite file')
    args = parser.parse_args()

    delete_ids = set()
    for delete_json in args.delete_jsons:
        with open(delete_json, 'r') as inf:
            delete_ids.update(json.loads(inf.read()))
    print(f'loaded {len(delete_ids)} pmids to remove from {args.output_database} from {len(args.delete_jsons)} files')
    con = sqlite3.connect(args.output_database)

    tables = [x[0] for x in con.execute("select name from sqlite_master where type='table'").fetchall()]
    print(f'have tables: {tables}')
    for t in tables:
        con.execute(f'delete from {t} where pmid in ({",".join(delete_ids)})')
    con.commit()

if __name__ == '__main__':
    main()
