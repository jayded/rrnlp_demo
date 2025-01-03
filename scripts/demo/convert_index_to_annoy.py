import itertools
import json
import sys
import faiss
import tqdm
#import hnswlib
import numpy as np
from annoy import AnnoyIndex

input_index_file = sys.argv[1]
input_json_file = sys.argv[2]
output_index_file = sys.argv[3]

with open(input_json_file, 'r') as inf:
    pmids = json.loads(inf.read())

source_index = faiss.read_index(input_index_file, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

print(f'loaded {len(pmids)} pmids')
embeddings = faiss.rev_swig_ptr(source_index.get_xb(), source_index.ntotal*source_index.d).reshape(source_index.ntotal, source_index.d)
print('source_index.d', source_index.d)
print('source_index.ntotal', source_index.ntotal)
print(f'loaded {len(embeddings)} embeddings')
#data = [
#    {"pmid": pmid, "vector": embedding}
#    for (pmid, embedding) in zip(pmids, embeddings)
#]

# prenormalized vectors. the angular distance measure is weird...
index = AnnoyIndex(source_index.d, metric='angular')
#index = AnnoyIndex(source_index.d, metric='dot')
print('insert')
for i in range(embeddings.shape[0]):
    index.add_item(i, embeddings[i])
print('trees')
index.build(n_trees=100)
print('save')

index.save(output_index_file)
#np.save(output_index_file, embeddings, allow_pickle=False)
#d = np.load(output_index_file, dtype='float32', mode='r', shape=(39423443, 768))
#assert d.size == embeddings.size
#assert np.allclose(embeddings, d)
