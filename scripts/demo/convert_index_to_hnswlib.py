import itertools
import json
import sys
import faiss
import tqdm
import hnswlib
import numpy as np

input_index_file = sys.argv[1]
input_json_file = sys.argv[2]
output_index_file = sys.argv[3]

with open(input_json_file, 'r') as inf:
    pmids = json.loads(inf.read())

source_index = faiss.read_index(input_index_file, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
target_index = hnswlib.Index(space='cosine', dim=source_index.d)
target_index.init_index(max_elements=source_index.ntotal, ef_construction=0, M=128)
target_index.set_num_threads(8)
#pmid_field = FieldSchema(
#    name='pmid',
#    dtype=DataType.VARCHAR,
#    is_primary=True,
#    max_length=20,
#)
#float_vector = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
#schema = CollectionSchema(fields=[pmid_field, float_vector], enable_dynamic_field=True)
#connections.connect()
#collection = Collection(output_index_file, schema=schema)
#client = MilvusClient(output_index_file)
#
#if client.has_collection(collection_name="pubmed_contriever"):
#    client.drop_collection(collection_name="pubmed_contriever")
#client.create_collection(
#    collection_name="pubmed_contriever",
#    schema=schema
##    fields=[
##        pmid_field,
##    ],
##    dimension=768,  # The vectors we will use in this demo has 768 dimensions
##    #enable_dynamic_field=False,
#)

assert len(pmids) == source_index.ntotal

print(f'loaded {len(pmids)} pmids')
embeddings = faiss.rev_swig_ptr(source_index.get_xb(), source_index.ntotal*source_index.d).reshape(source_index.ntotal, source_index.d)
print(f'loaded {len(embeddings)} embeddings')
#data = [
#    {"pmid": pmid, "vector": embedding}
#    for (pmid, embedding) in zip(pmids, embeddings)
#]

partitions = 40000
#for pmid, embedding in tqdm.tqdm(zip(pmids, embeddings), total=len(pmids)):
for elems in tqdm.tqdm(itertools.batched(embeddings, n = partitions), total=len(pmids) // partitions):
    target_index.add_items(elems)
    #data = [{
    #    'pmid': pmid,
    #    'embedding': embedding,
    #} for (pmid, embedding) in elems]
    ##data = [{
    ##    'pmid': pmid,
    ##    'embedding': embedding,
    ##}]
    ##res = client.insert(collection_name="pubmed_contriever", data=data)
    ##collection.insert(data)
    #res = client.insert(collection_name="pubmed_contriever", data=data)
target_index.save_index(output_index_file)
