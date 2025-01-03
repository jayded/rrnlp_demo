import itertools
import json
import sys
import faiss
import tqdm
#import hnswlib
import numpy as np

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

np.save(output_index_file, embeddings, allow_pickle=False)
d = np.load(output_index_file, dtype='float32', mode='r', shape=(39423443, 768))
assert d.size == embeddings.size
assert np.allclose(embeddings, d)
