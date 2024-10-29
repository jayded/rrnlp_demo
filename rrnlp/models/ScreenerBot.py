import copy
import json
import os
import sys
import time
from typing import List, Optional, Tuple, Type

#import faiss
import numpy as np

#from pymilvus import MilvusClient
import torch
import torch.nn.functional as F

from annoy import AnnoyIndex

from transformers import AutoModel, AutoTokenizer

from rrnlp.models import get_device

from info_nce import InfoNCE

#weights_path = rrnlp.models.weights_path
# TODO
#weights = 'facebook/contriever'
#embeddings_path = ''
#embeddings_path = '/media/more_data/jay/overflow/ei_demo/data/contriever_index/faiss_index.bin'
#pmids_positions = '/media/more_data/jay/overflow/ei_demo/data/contriever_index/pmids.json'

def load_screener(
        weights: str=None, #hf model string or path
        #embeddings_path: str=embeddings_path, # milvus embedding path
        #milvus_collection: str='contriever', #
        embeddings_path: str=None, #Annoy embedding path
        pmids_positions: str=None, #turn a faiss index to a pmid
        device='auto',
    ):
    device = get_device(device)
    model = AutoModel.from_pretrained(weights).to(device)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    #with open(embeddings_path, 'rb') as inf:
    #    reader = faiss.PyCallbackIOReader(inf.read)
    #    index = faiss.read_index(reader, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    #    #index = faiss.read_index(reader)
    #with open(pmids_positions, 'r') as inf:
    #    position_to_pmid = json.loads(inf.read())
    ranker = AnnoyRanker.create(embeddings_path, pmids_positions)
    return Screener(model, tokenizer, ranker) #, position_to_pmid)


# TODO this probably also needs to be factored out so we can have a faiss version too
# since these indexes are such a pain and don't always cooperate with being loaded in
# different systems/RAM configurations.
class AnnoyRanker:
    def __init__(
        self,
        index: AnnoyIndex,
        position_to_pmid: List,
        # TODO: grow a metric?
    ):
        self.index = index
        self.position_to_pmid = position_to_pmid
        self.pmid_to_position = dict(reversed(x) for x in enumerate(self.position_to_pmid))
        assert len(self.position_to_pmid) == self.index.get_n_items()

    @classmethod
    def create(cls, index_path, pmids_path):
        # TODO is using dot sacrificing anything?
        index = AnnoyIndex(768, 'dot')
        index.load(index_path)
        with open(pmids_path, 'r') as inf:
            position_to_pmid = json.loads(inf.read())
        return AnnoyRanker(index, position_to_pmid)

    def compute_distances(self, vec: np.array, pmids: List):
        vec = np.array(vec)
        vec = vec.reshape(-1, 1)
        positions = [self.pmid_to_position[pmid] for pmid in pmids]
        # TODO note this uses dot, if we want a different measure (say, angular), we'll need to compute that manually since it seems Annoy doesn't expose that (note: verify this)
        embeds = [self.index.get_item_vector(position) for position in positions]
        embeds = np.array(embeds)
        distances = embeds @ vec
        distances = distances.squeeze()
        return pmids, distances

    def get_pmid_embeds(self, pmids: List):
        positions = [self.pmid_to_position[pmid] for pmid in pmids]
        embeds = [self.index.get_item_vector(position) for position in positions]
        embeds = np.array(embeds)
        return embeds

    def find_nns_for_vec(self, vec: np.array, pmids: Optional[List]=None):
        raise NotImplementedError("Not presently using this capacity of a vector index")

    def known_pmids(self):
        return set(self.pmid_to_position.keys())

class Screener:
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        #position_to_pmid: List,
        ranker: AnnoyRanker,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ranker = ranker
        #self.position_to_pmid = position_to_pmid
        # note: a pmid might appear multiple times, this takes the _last_ one
        # as later entries overwrite earlier ones in the dict constructor
        #self.pmid_to_position = dict(reversed(x) for x in enumerate(self.position_to_pmid))

    @classmethod
    def mean_pooling(cls, token_embeddings, mask):
        # https://huggingface.co/facebook/contriever
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def predict_for_topic(self, topic: str, pmids: List, custom_model_path: Optional[str]=None):
        pmids = set(pmids)
        pmids_in_db = list(self.ranker.known_pmids() & pmids) # list so the order is consistent
        print(f'Processing for {len(pmids)} pmids')
        start_time = time.time()
        input_embeddings = self.embed_topic(topic, custom_model_path=custom_model_path)
        embeddings_compute_end = time.time()
        print(f'Computing topic embeddings took {embeddings_compute_end - start_time}')

        if pmids is not None and len(pmids) > 0:
            print(f'Have {len(pmids_in_db)} pmids of total requested {len(pmids)} pmids')
            res_pmids, distances = self.ranker.compute_distances(input_embeddings, pmids_in_db)
            #filter_ids = [self.pmid_to_position[pmid] for pmid in pmids_in_db]
            #id_selector = faiss.IDSelectorArray(filter_ids)
            #params = faiss.SearchParameters(sel=id_selector)
            #distance, positions = self.embedding_index.search(input_embeddings, len(pmids_in_db), params=params)
            #res = zip((self.position_to_pmid[x] for x in positions), distance)
            embedding_end = time.time()
            print(f'Computing distances took {embedding_end - embeddings_compute_end} seconds')
        else:
            assert False
        # todo handle zero-length lists
        res = list(zip(res_pmids, distances))
        sorted(res, key=lambda x: x[1], reverse=True)
        return res

    def embed_topic(self, topic: str, custom_model_path=None):
        if custom_model_path is not None:
            model = AutoModel.from_pretrained(custom_model_path)
        else:
            model = self.model
        inputs = self.tokenizer(topic, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        print(inputs)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            embeds = Screener.mean_pooling(token_embeddings=outputs['last_hidden_state'], mask=inputs['attention_mask'])
            embeds = F.normalize(embeds, p=2, dim=1)
        return embeds.detach()

    def embed_abs(self, ti_abs: List[dict]):
        input_texts = [ti_ab['ti'] + '\n' + ti_ab['ab'] for ti_ab in ti_abs]
        inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        outputs = self.model(inputs, return_dict=True)
        embeds = Screener.mean_pooling(token_embeddings=outputs['last_hidden_state'], mask=inputs['attention_mask'])
        embeds = F.normalize(embeds, p=2, dim=1)
        return embeds.detach()

    def finetune_for_annotations(
            self,
            topic: str,
            positive_ids: List,
            negative_ids: List,
            epochs=10,
            lr=5e-5,
            finetune_only_last_layer=False,
            # for the loss function
            clone_model=True,
        ):
        # I reluctantly use HF code here
        # however, I am rolling my own training because I don't expect the APIs to be stable.
        if clone_model:
            model = copy.deepcopy(self.model)
        else:
            model = self.model
        model = self.model
        model.train()
        if finetune_only_last_layer:
            for param in model.parameters():
                param.requires_grad = False
            # finetune only the last layer
            for param in model.encoder.layer[-1].parameters():
                param.requires_grad = True

        positive_ids_ = list(self.ranker.known_pmids() & set(positive_ids))
        negative_ids_ = list(self.ranker.known_pmids() & set(negative_ids))
        positive_embeddings = torch.tensor(self.ranker.get_pmid_embeds(positive_ids_), dtype=torch.float)
        negative_embeddings = torch.tensor(self.ranker.get_pmid_embeds(negative_ids_), dtype=torch.float)
        print(f'Given {len(positive_ids)} positive ids, know about {len(positive_ids_)}')
        print(f'Given {len(negative_ids)} negative ids, know about {len(negative_ids_)}')

        training_params = list(filter(lambda x: x.requires_grad, model.parameters()))
        print(f'Training {len(training_params)} parameters')
        optimizer = torch.optim.AdamW(training_params, lr=lr)

        # tokenize once to save the computer
        topic_inputs = self.tokenizer([topic], padding=True, truncation=True, return_tensors="pt").to(self.model.device)

        # msmarco fine-tuning steps
        # adamw
        # lr 1e-4
        # 20k gradient steps
        # bs 1024
        # warmup 1k steps
        # temperature 0.05 (with contrastive loss)
        # InfoNCE loss:

        def eval_ish(topic_embed):
            all_docs = torch.cat([positive_embeddings, negative_embeddings], dim=0)
            labels = torch.LongTensor([1 for _ in positive_embeddings] + [0 for _ in negative_embeddings])
            all_evals = (topic_embed @ all_docs.t()).squeeze()
            sort_idx = torch.argsort(all_evals, descending=True)
            sort_labels = labels[sort_idx]
            aps = []
            mrr = None
            dcgs_standard = []
            idcgs = [] # idealized dcg
            # the indexes for the dcgs begin at one
            for i, label in enumerate(sort_labels):
                ap = torch.sum(sort_labels[:i+1]) / (i + 1)
                aps.append(ap)
                if label == 1 and mrr is None:
                    mrr = 1 / (i + 1)
                dcgs_standard.append(label/torch.log2(torch.tensor(i + 1 + 1)))
                idcgs.append((torch.pow(2, label) - 1) / torch.log2(torch.tensor(i + 1 + 1)))

            ap = np.mean(aps)
            pat5 = sort_labels[:5].sum() / 5
            pat1 = int(sort_labels[0] == 1)
            idealized_dcg = np.mean(idcgs)
            dcgs_standard = np.mean(dcgs_standard)

            return {
                'ap': float(ap),
                'mrr': float(mrr),
                'precision at 1': float(pat1),
                'precision at 5': float(pat5.item()),
                'idcg': float(idealized_dcg),
                'dcg': float(dcgs_standard.item()),
            }

        # TODO: pair this?
        loss_fn = InfoNCE(negative_mode='unpaired')
        losses = []
        metrics = [{} for _ in range(epochs+1)]
        # TODO batches? this is _one_ instance
        for epoch in range(epochs):
            optimizer.zero_grad()
            topic_embeds = model(**topic_inputs, return_dict=True)
            topic_embeds = topic_embeds['pooler_output']
            topic_embeds = F.normalize(topic_embeds, p=2, dim=1)
            res = eval_ish(topic_embeds.detach())
            loss = loss_fn(topic_embeds.squeeze().repeat(positive_embeddings.size()[0], 1), positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
            metrics[epoch].update(res)
            metrics[epoch+1]['loss'] = loss.item()
            print(f'epoch {epoch-1}', res)
            print(f'epoch {epoch} loss {loss.detach().cpu()}')
            losses.append(loss.detach().cpu().item())

        with torch.no_grad():
            topic_embeds = model(**topic_inputs, return_dict=True)
            topic_embeds = topic_embeds['pooler_output']
            res = eval_ish(topic_embeds)
            metrics[-1].update(res)
            print(f'epoch {epochs}', res)

        model.eval()
        return model, metrics


