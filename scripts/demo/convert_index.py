import itertools
import json
import sys
import faiss
import tqdm
from pymilvus import connections, Collection
from pymilvus import DataType
from pymilvus import MilvusClient
from pymilvus import CollectionSchema, FieldSchema


input_index_file = sys.argv[1]
input_json_file = sys.argv[2]
output_index_file = sys.argv[3]

with open(input_json_file, 'r') as inf:
    pmids = json.loads(inf.read())

index = faiss.read_index(input_index_file, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
pmid_field = FieldSchema(
    name='pmid',
    dtype=DataType.VARCHAR,
    is_primary=True,
    max_length=20,
)
float_vector = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
schema = CollectionSchema(fields=[pmid_field, float_vector], enable_dynamic_field=True)
#connections.connect()
#collection = Collection(output_index_file, schema=schema)
client = MilvusClient(output_index_file)
#
if client.has_collection(collection_name="pubmed_contriever"):
    client.drop_collection(collection_name="pubmed_contriever")
client.create_collection(
    collection_name="pubmed_contriever",
    schema=schema
#    fields=[
#        pmid_field,
#    ],
#    dimension=768,  # The vectors we will use in this demo has 768 dimensions
#    #enable_dynamic_field=False,
)

assert len(pmids) == index.ntotal

print(f'loaded {len(pmids)} pmids')
embeddings = faiss.rev_swig_ptr(index.get_xb(), index.ntotal*index.d).reshape(index.ntotal, index.d)
print(f'loaded {len(embeddings)} embeddings')
#data = [
#    {"pmid": pmid, "vector": embedding}
#    for (pmid, embedding) in zip(pmids, embeddings)
#]

partitions = 40000
#for pmid, embedding in tqdm.tqdm(zip(pmids, embeddings), total=len(pmids)):
for elems in tqdm.tqdm(itertools.batched(zip(pmids, embeddings), n = partitions), total=len(pmids) // partitions):
    data = [{
        'pmid': pmid,
        'embedding': embedding,
    } for (pmid, embedding) in elems]
    #data = [{
    #    'pmid': pmid,
    #    'embedding': embedding,
    #}]
    #res = client.insert(collection_name="pubmed_contriever", data=data)
    #collection.insert(data)
    res = client.insert(collection_name="pubmed_contriever", data=data)
