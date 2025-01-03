import argparse
import glob
import json
import os

import warnings
warnings.filterwarnings("ignore")

import faiss


def main():
    parser = argparse.ArgumentParser(description='decode flan over a csv of inputs')
    parser.add_argument('--input_indexes', nargs='+', required=True, help='compressed pubmed downloaded nxml to be read')
    parser.add_argument('--output_index', required=True, help='how many title/abstracts to embed at once')
    parser.add_argument('--input_json_indexes', nargs='+', required=True, help='list of pubmed ids/positions')
    parser.add_argument('--output_json_index', required=True, help='where do the embeds get stored')
    args = parser.parse_args()

    #master_index = faiss.IndexFlatIP(model.config.hidden_size) #, faiss.METRIC_INNER_PRODUCT)
    master_index = None
    master_pmids_list = []
    assert len(args.input_json_indexes) == len(args.input_indexes)
    for json_index, faiss_index in zip(args.input_json_indexes, args.input_indexes):
        print(json_index)
        print(faiss_index)
        with open(json_index, 'r') as inf:
            pmids = json.loads(inf.read())
        with open(faiss_index, 'rb') as inf:
            # https://github.com/facebookresearch/faiss/blob/a17a631/tests/test_io.py#L125
            reader = faiss.PyCallbackIOReader(inf.read)
            index = faiss.read_index(reader)
        assert index.ntotal == len(pmids)
        if master_index is None:
            master_index = index
        else:
            master_index.merge_from(index)
        master_pmids_list.extend(pmids)
        assert master_index.ntotal == len(master_pmids_list)
        print(master_index.ntotal, len(master_pmids_list))
        faiss.write_index(master_index, args.output_index)
        with open(args.output_json_index, 'w') as of:
            of.write(json.dumps(master_pmids_list))
        print(os.path.getsize(args.output_index) // 100000000 )


    #faiss.write_index(index, args.output_index)
    #with open(args.output_json_index, 'w') as of:
    #    of.write(json.dumps(master_pmids_list))

if __name__ == "__main__":
    main()
