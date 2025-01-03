import glob

import pubmed_parser as pp

files = list(glob.glob('/scratch/deyoung.j/pubmed/*xml.gz')) + list(glob.glob('/scratch/deyoung.j/pubmed_updates/*xml.gz'))

total = 0
for f in files:
    f_count = 0
    for nxml in pp.parse_medline_xml(f):
        f_count += 1
        total += 1
    print(f_count, f)
print(total, 'total')    
