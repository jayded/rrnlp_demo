import argparse
import itertools
import json
import sys
import tqdm
import numpy as np
#from annoy import AnnoyIndex
from voyager import Index, Space

def main():
    parser = argparse.ArgumentParser(description='decode flan over a csv of inputs')
    #parser.add_argument('--input_gzs', nargs='+', required=True, help='compressed pubmed downloaded nxml to be read')
    #parser.add_argument('--batch_size', default=8, type=int, help='how many title/abstracts to embed at once')
    parser.add_argument('--json_index', required=True, nargs='+', help='list of pubmed ids/positions (assumed to be in ascending order of update)')
    parser.add_argument('--np_embeds', required=True,  nargs='+', help='numpy embeds files (assumed to correspond to the json_index)')
    parser.add_argument('--output_index_file', required=True, help='output annoy index')
    parser.add_argument('--output_json_index_file', required=True, help='output annoy index')
    args = parser.parse_args()

    assert len(args.json_index) == len(args.np_embeds)
    print(f'creating index')
    index = Index(Space.InnerProduct, num_dimensions=768, max_elements=50000000)
    #index = AnnoyIndex(768, metric='dot')
    #index.on_disk_build(args.output_index_file)

    print('reading pmids')
    # find the latest version of any article and embed only that one
    pmid_to_input_file = dict()
    for json_index_file, np_embeds_file in zip(args.json_index, args.np_embeds):
        with open(json_index_file, 'r') as inf:
            contents = json.loads(inf.read())
            for pmid in contents:
                pmid_to_input_file[pmid] = np_embeds_file

    
    print('reading embeds')
    all_pmids = []
    for json_index_file, np_embeds_file in tqdm.tqdm(list(zip(args.json_index, args.np_embeds))):
        with open(json_index_file, 'r') as inf:
            contents = json.loads(inf.read())
        embeds = np.load(np_embeds_file)
        for pmid, embed in zip(contents, embeds):
            if pmid_to_input_file[pmid] == np_embeds_file:
                pos = index.add_item(embed)
                assert pos == len(all_pmids)
                all_pmids.append(pmid)

    print('trees')
    index.build(n_trees=100)
    print('save')

    index.save(args.output_index_file)
    with open(args.output_json_index_file, 'w') as of:
        of.write(json.dumps(all_pmids))

#input_index_file = sys.argv[1]
#input_json_file = sys.argv[2]
#output_index_file = sys.argv[3]
#
#with open(input_json_file, 'r') as inf:
#    pmids = json.loads(inf.read())
#
#source_index = faiss.read_index(input_index_file, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
#
#print(f'loaded {len(pmids)} pmids')
#embeddings = faiss.rev_swig_ptr(source_index.get_xb(), source_index.ntotal*source_index.d).reshape(source_index.ntotal, source_index.d)
#print('source_index.d', source_index.d)
#print('source_index.ntotal', source_index.ntotal)
#print(f'loaded {len(embeddings)} embeddings')
##data = [
##    {"pmid": pmid, "vector": embedding}
##    for (pmid, embedding) in zip(pmids, embeddings)
##]
#
## prenormalized vectors. the angular distance measure is weird...
#index = AnnoyIndex(source_index.d, metric='angular')
##index = AnnoyIndex(source_index.d, metric='dot')
#print('insert')
#for i in range(embeddings.shape[0]):
#    index.add_item(i, embeddings[i])
#print('trees')
#index.build(n_trees=100)
#print('save')
#
#index.save(output_index_file)
##np.save(output_index_file, embeddings, allow_pickle=False)
##d = np.load(output_index_file, dtype='float32', mode='r', shape=(39423443, 768))
##assert d.size == embeddings.size
##assert np.allclose(embeddings, d)
#

if __name__ == "__main__":
    main()

