import argparse
import glob
import json
import os

import warnings

import pandas as pd
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='merge a title and abstract set')
    parser.add_argument('--input_jsons', nargs='+', required=True, help='dict of {"title": pmid -> title, "abstract": pmid -> abstract}')
    parser.add_argument('--output_json', required=True, help='where does the combine object get stored')
    parser.add_argument('--output_df', required=True, help='where does the output dataframe get stored')
    args = parser.parse_args()

    #master_index = faiss.IndexFlatIP(model.config.hidden_size) #, faiss.METRIC_INNER_PRODUCT)
    all_jsons = None
    input_jsons = []
    rows = []
    for json_index_file in args.input_jsons:
        if '*' in json_index_file:
            input_jsons.extend(glob.glob(json_index_file))
        else:
            input_jsons.append(json_index_file)
    input_jsons.sort()
    for json_index_file in input_jsons:
        print(json_index_file)
        with open(json_index_file, 'r') as inf:
             json_index = json.loads(inf.read())
        if all_jsons is None:
            all_jsons = json_index
        else:
            assert set(all_jsons.keys()) == set(json_index.keys())
            for k in all_jsons.keys():
                all_jsons[k].update(json_index[k])
        any_key = list(json_index.keys())[0]
        for pmid in json_index[any_key].keys():
            row = {'pmid': pmid}
            for k in json_index.keys():
                row[k] = json_index[k][pmid]
            rows.append(row)

    with open(args.output_json, 'w') as of:
        of.write(json.dumps(all_jsons))

    df = pd.DataFrame(rows)
    if args.output_df.endswith('parquet'):
        df.to_parquet(args.output_df)
    elif args.output_df.endswith('csv'):
        df.to_csv(args.output_df)
    else:
        assert False, 'do not know how to write to this file: ' + args.output_df

if __name__ == "__main__":
    main()

