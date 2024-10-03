import json
import sys

import pandas as pd

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as inf:
   field_to_pmid_to_value = json.loads(inf.read())

fields = list(field_to_pmid_to_value.keys())
pmids = field_to_pmid_to_value[fields[0]].keys()

elems = []
for pmid in pmids:
    res = {k:field_to_pmid_to_value[k][pmid] for k in fields}
    res['pmid'] = pmid
    elems.append(res)

df = pd.DataFrame(elems)
df.to_csv(output_file)
