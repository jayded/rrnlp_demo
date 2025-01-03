import argparse
import glob
import json
import os

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from typing import List

#import faiss
import pubmed_parser as pp
import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def mean_pooling(token_embeddings, mask):
    # https://huggingface.co/facebook/contriever
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def embed_abs(tokenizer, model, ti_abs: List[dict]):
    input_texts = [ti_ab['ti'] + '\n' + ti_ab['ab'] for ti_ab in ti_abs]
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        embeds = outputs['pooler_output']
        #embeds = mean_pooling(outputs['last_hidden_state'], inputs['attention_mask'])
        #embeds = F.normalize(embeds, p=2, dim=1)
    return embeds.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='decode flan over a csv of inputs')
    parser.add_argument('--input_gzs', nargs='+', required=True, help='compressed pubmed downloaded nxml to be read')
    parser.add_argument('--batch_size', default=8, type=int, help='how many title/abstracts to embed at once')
    parser.add_argument('--json_index', required=True, help='list of pubmed ids/positions')
    parser.add_argument('--np_embeds', required=True, help='file to store numpy embeds')
    #parser.add_argument('--faiss_index', required=True, help='where do the embeds get stored')
    parser.add_argument('--model', required=True, help='where do the embeds get stored')
    args = parser.parse_args()

    assert False, "convert this to annoy index!"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).cuda()

    if os.path.exists(args.json_index):
        with open(args.json_index, 'r') as inf:
            pmids = json.loads(inf.read())
        print(f'loaded {len(pmids)} pmids')
    else:
        pmids = []
        with open(args.json_index, 'w') as of:
            of.write(json.dumps(pmids))
    processed_pmids = set(pmids)

    #if os.path.exists(args.faiss_index):
    #    with open(args.faiss_index, 'rb') as inf:
    #        # https://github.com/facebookresearch/faiss/blob/a17a631/tests/test_io.py#L125
    #        reader = faiss.PyCallbackIOReader(inf.read)
    #        index = faiss.read_index(reader)
    #    print(f'loaded existing index from {args.faiss_index} with {index.ntotal} embeds')
    #else:
    #    # TODO should fetch this dim from the model
    #    index = faiss.IndexFlatIP(model.config.hidden_size) #, faiss.METRIC_INNER_PRODUCT)
    #    faiss.write_index(index, args.faiss_index)

    #assert index.ntotal == len(pmids)

    processed_total = 0
    processed_batches = 0
    all_embeds = []
    all_pmids = []
    for gz_file in args.input_gzs:
        print(f'processing {gz_file}')
        current_batch = []
        skipped_elems = []
        for i, article_info in enumerate(pp.parse_medline_xml(gz_file)):
            title = article_info['title']
            if 'Erratum' in title:
                skipped_elems.append(i)
                continue
            abstract = article_info['abstract']
            pmid = article_info['pmid']
            if pmid in processed_pmids:
                skipped_elems.append(i)
                continue
            if abstract is None or len(abstract.strip()) == 0:
                continue

            current_batch.append({
                'ti': title,
                'ab': abstract,
                'pmid': pmid,
            })
            all_pmids.append(pmid)

            if len(current_batch) == args.batch_size:
                embeds = embed_abs(tokenizer, model, current_batch)
                all_embeds.extend(embeds)
                processed_total += len(current_batch)
                processed_batches += 1
                batch_pmids = [elem['pmid'] for elem in current_batch]
                pmids.extend(batch_pmids)
                assert len(pmids) == index.ntotal
                current_batch = []
                if processed_batches % 100 == 0:
                    print('writing index')
                    os.rename(args.json_index, args.json_index + '.bak')
                    with open(args.json_index, 'w') as of:
                        of.write(json.dumps(pmids))
                    os.remove(args.json_index + '.bak')

                    #os.rename(args.faiss_index, args.faiss_index + '.bak')
                    #faiss.write_index(index, args.faiss_index)
                    #os.remove(args.faiss_index + '.bak')
                    #processed_pmids.update(set(batch_pmids))

        print(f'skipped index: {len(skipped_elems)}')
        if len(current_batch) > 0:
            embeds = embed_abs(tokenizer, model, current_batch)
            all_embeds.extend(embeds)
            batch_pmids = [elem['pmid'] for elem in current_batch]
            pmids.extend(batch_pmids)
            processed_total += len(current_batch)
            processed_batches += 1
            current_batch = []
        #os.rename(args.json_index, args.json_index + '.bak')
        with open(args.json_index, 'w') as of:
            of.write(json.dumps(pmids))
        #os.remove(args.json_index + '.bak')

        #os.rename(args.faiss_index, args.faiss_index + '.bak')
        #faiss.write_index(index, args.faiss_index)
        #os.remove(args.faiss_index + '.bak')
            
    #os.rename(args.json_index, args.json_index + '.bak')
    #with open(args.json_index, 'w') as of:
    #    of.write(json.dumps(pmids))
    #os.remove(args.json_index + '.bak')

    #os.rename(args.faiss_index, args.faiss_index + '.bak')
    #faiss.write_index(index, args.faiss_index)
    #os.remove(args.faiss_index + '.bak')


if __name__ == "__main__":
    main()


