import copy
import json
import os
import sys 
from typing import List, Optional, Tuple, Type

import faiss
import numpy as np 

#from pymilvus import MilvusClient
import torch 
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from rrnlp.models import get_device

#weights_path = rrnlp.models.weights_path
# TODO
weights = 'facebook/contriever'
#embeddings_path = ''
embeddings_path = '/media/more_data/jay/overflow/ei_demo/data/contriever_index/faiss_index.bin'
pmids_positions = '/media/more_data/jay/overflow/ei_demo/data/contriever_index/pmids.json'

def load_screener(
        weights: str=weights, #hf model string or path
        #embeddings_path: str=embeddings_path, # milvus embedding path
        #milvus_collection: str='contriever', # 
        embeddings_path: str=embeddings_path, #faiss embedding path
        pmids_positions: str=pmids_positions, #turn a faiss index to a pmid
        device='auto',
    ):
    device = get_device(device)
    model = AutoModel.from_pretrained(weights).to(device)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    #milvus_client = client = MilvusClient(embeddings_path)
    # TODO should these items be joined together?
    index = None
    #with open(embeddings_path, 'rb') as inf:
    #    reader = faiss.PyCallbackIOReader(inf.read)
    #    index = faiss.read_index(reader, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    #    #index = faiss.read_index(reader)
    with open(pmids_positions, 'r') as inf:
        position_to_pmid = json.loads(inf)
    return Screener(model, tokenizer, index, position_to_pmid)


class Screener:
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        embedding_index, # what is the right type?
        #collection_name, # TODO is this the right choice? do we want to add other options?
        position_to_pmid: List,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_index = embedding_index
        self.position_to_pmid = position_to_pmid
        # note: a pmid might appear multiple times, this takes the _last_ one 
        # as later entries overwrite earlier ones in the dict constructor
        self.pmid_to_position = dict(reversed(x) for x in enumerate(self.position_to_pmid))

    def mean_pooling(cls, token_embeddings, mask):
        # https://huggingface.co/facebook/contriever
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings.detach().cpu().numpy()

    def predict_for_topic(self, topic: str, pmids: List, use_disk_location: Optional[str]=None):
        pmids = set(pmids)
        input_embeddings = self.embed_topic(topic, use_disk_location=use_disk_location)

        if pmids is not None and len(pmids) > 0:
            pmids = ','.join(set(pmids))
            res = client.search(
                collection_name=self.collection_name,
                data=input_embeddings,
                filter=f"pmid in [pmids]",
                limit=len(pmids),
                output_fields=["pmid"],
            )
            filter_ids = [self.pmid_to_position[pmid] for pmid in pmids]
            id_selector = faiss.IDSelectorArray(filter_ids)
            params = faiss.SearchParameters(sel=id_selector)
            distance, positions = self.embedding_index.search(input_embeddings, len(pmids), params=params)
        else:
            assert False
            #params = None
            #distance, positions = self.embedding_index.search(input_embeddings, 10000, params=params)
        res = zip((self.position_to_pmid[x] for x in positions), distance)
        #res = zip((self.position_to_pmid[x] for x in positions), distance, positions)
        #if pmids is not None:
        #    res = filter(lambda x: x in pmids, res)
        #res = list(res)
        return res

    def embed_topic(self, topic: str, use_disk_location=None):
        if use_disk_location is not None:
            model = AutoModel.from_pretrained(use_disk_location)
        else:
            model = self.model
        inputs = self.tokenizer([topic], padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = model(inputs, output_dict=True)
            embeds = Screener.mean_pooling(outputs['return_attention_mask'], inputs['return_attention_mask'])
            embeds = F.normalize(embeds, p=2, dim=1)
        return embeds.detach()

    def embed_abs(self, ti_abs: List[dict]):
        input_texts = [ti_ab['ti'] + '\n' + ti_ab['ab'] for ti_ab in ti_abs]
        inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        outputs = self.model(inputs, output_dict=True)
        embeds = Screener.mean_pooling(outputs['return_attention_mask'], inputs['return_attention_mask'])
        embeds = F.normalize(embeds, p=2, dim=1)
        return embeds.detach()

    def finetune_for_annotations(
            self,
            topic: str,
            positive_ids: List,
            negative_ids: List,
            epochs=5,
            lr=1e-4,
            finetune_only_last_layer=True,
            # for the loss function
            temperature=0.05,
            clone_model=True,
        ):
        # I reluctantly use HF code here
        # however, I am rolling my own training because I don't expect the APIs to be stable.
        if clone_model:
            model = copy.deepcopy(self.model)
        else:
            model = self.model
        if finetune_only_last_layer:
            for param in model.parameters():
                param.requires_grad = False
            # finetune only the last layer
            for param in model.encoder.layer[-1].parameters():
                param.requires_grad = True
        
        positive_embeddings = [self.embedding_index.xb.at(self.pmid_to_position[pmid]) for pmid in positive_ids]
        negative_embeddings = [self.embedding_index.xb.at(self.pmid_to_position[pmid]) for pmid in negative_ids]

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
        def loss_fn(topic_embedding, positive_doc_embeddings, negative_doc_embeddings, temperature):
            def s(topic_embedding, doc):
                return topic_embedding.dot(doc)
            negative_loss_components = []
            for negative_doc_embedding in negative_doc_embeddings:
                negative_loss_components.append(F.exp(s(topic_embedding, negative_doc_embedding)/temperature))
            negative_loss_component = torch.sum(negative_loss_components)

            loss_components = []
            
            for positive_doc_embedding in positive_doc_embeddings:
                pos_value = s(topic_embedding, positive_doc_embedding)/temperature
                loss_components.append(pos_value / (pos_value + negative_loss_component))

            loss = -1 * torch.sum(loss_components)
            return loss

        losses = []
        for epoch in epochs:
            optimizer.zero_grad()
            topic_embeds = self.model(inputs, output_dict=True)
            topic_embeds = Screener.mean_pooling(topic_embeds['return_attention_mask'], topic_embeds['return_attention_mask'])
            topic_embeds = F.normalize(topic_embeds, p=2, dim=1)
            loss = loss_fn(topic_embeds, positive_embeddings, negative_embeddings, temperature)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu())

        return model, losses


