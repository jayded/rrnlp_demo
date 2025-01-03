import argparse
import glob
import json
import os

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

import pubmed_parser as pp



def main():
    parser = argparse.ArgumentParser(description='decode flan over a csv of inputs')
    parser.add_argument('--input_gzs', nargs='+', required=True, help='compressed pubmed downloaded nxml to be read')
    parser.add_argument('--output_json', required=True, help='output json file with title and abstract fields')
    args = parser.parse_args()

    if os.path.exists(args.output_json):
        with open(args.output_json, 'r') as inf:
            title_abstracts = json.loads(inf.read())
    else:
        title_abstracts = {
            'titles': dict(),
            'abstracts': dict(),
            'years': dict(),
            #'file': dict()
        }

    for gz_file in args.input_gzs:
        print(f'processing {gz_file}')
        for i, article_info in enumerate(pp.parse_medline_xml(gz_file)):
            title = article_info['title']
            if 'Erratum' in title:
                continue
            abstract = article_info['abstract']
            pmid = article_info['pmid']
            title_abstracts['titles'][pmid] = title
            title_abstracts['abstracts'][pmid] = abstract
            title_abstracts['years'][pmid] = article_info['pubdate']
            #title_abstracts['file'][pmid] = gz_file
            if i % 500 == 0:
                if os.path.exists(args.output_json):
                    os.rename(args.output_json, args.output_json + '.bak')
                with open(args.output_json, 'w') as of:
                    of.write(json.dumps(title_abstracts))

    os.rename(args.output_json, args.output_json + '.bak')
    with open(args.output_json, 'w') as of:
        of.write(json.dumps(title_abstracts))

if __name__ == "__main__":
    main()


