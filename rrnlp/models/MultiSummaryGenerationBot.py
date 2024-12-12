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

from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rrnlp.models import get_device

#weights_path = rrnlp.models.weights_path
# TODO
weights = '/home/ubuntu/rrnlp-demo/flan_t5_large_adamw_hf_steps10000_lr1e-4'
default_instruct_file = os.path.join(os.path.dirname(__file__), 'MultiSummaryGenerationExample.json')

def load_openai_summary_bot(
        api_key_file=None,
        api_key=None,
        instruct_file=default_instruct_file,
        instruct_studies: List[str]=None,
        instruct_target: str=None
    ):
    assert (api_key_file is None) != (api_key is None), f'Exactly one of api_key_file or api_key must be provided, have api_key {api_key} and api_key_file {api_key_file}'
    assert (instruct_studies is None) == (instruct_target is None), 'Must provide both instruct examples and a target, xor an input file containing both!'
    assert (instruct_file is None) != (instruct_studies is None), 'Must not provide both an instruct file and studies!'

    if instruct_file is not None:
        with open(instruct_file, 'r') as inf:
            instruct_contents = json.loads(inf.read())
            instruct_studies = instruct_contents['studies']
            instruct_target = instruct_contents['target']

    if api_key_file is not None:
        with open(api_key_file, 'r') as inf:
            api_key = inf.read().strip()
    return OpenAIMDSBot(
        api_key = api_key,
        gpt_model = 'gpt-4o',
        instruct_studies = instruct_studies,
        instruct_target = instruct_target,
    )


class OpenAIMDSBot:

    def __init__(
        self,
        api_key,
        gpt_model,
        instruct_studies: List[str],
        instruct_target: str,
    ):
        self.client = OpenAI(api_key=api_key)
        self.gpt_model = gpt_model
        self.instruct_studies = instruct_studies
        self.instruct_target = instruct_target

    def predict_for_docs(self, topic: str, docs: List[str]) -> (str, dict):
        query = [
                    # worked well for gpt-4-0613
                    {
                        'role': 'system',
                        'content': 'You are a systematic reviewing expert. Your job is to read randomized control trial reports and assist a medical researcher. You will aid in drafting systematic reviews.'
                    },
                    {
                        'role': 'user',
                        'content': 'Please provide a draft systematic review for the studies below: {studies}. \nStart with the conclusions of the review only, a more detailed analysis will happen later.'.format(studies='\n'.join([f'Study: {study}' for study in self.instruct_studies]))
                    },
                    {
                        'role': 'assistant',
                        'content': self.instruct_target,
                    },
                    {
                        'role': 'user',
                        'content': 'Please provide a draft systematic review for the studies below: {studies}. \nStart with the conclusions of the review only, a more detailed analysis will happen later.'
                    },
                ]
        query[-1]['content'] = query[-1]['content'].format(studies='\n'.join([f'Study: {study}' for study in docs]))

        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=query,
            temperature=0,
            max_tokens=500,
        )

        print(type(response), dir(response))
        print(response.to_json())
        response = response.to_dict()
        print(response)
        completion = response['choices'][-1]['message']['content']

        return completion, response
        #return response['choices'][-1]['message']['content'], response

def load_mds_summary_bot(
        weights: str=weights, #hf model string or path
        device='auto',
    ):
    device = get_device(device)
    model = AutoModelForSeq2SeqLM.from_pretrained(weights).to(device)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    return MDSSummaryBot(model, tokenizer)


class MDSSummaryBot:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def predict_for_topic(self, topic: str, pmids: List):
        raise not NotImplemented()

    def predict_for_docs(self, topic: str, docs: List[str]):
        input_ids, attention_mask = self._prep(topic, docs)
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, max_length=512)
        print(outputs)
        outputs = [self.tokenizer.decode(pred.cpu(), skip_special_tokens=True) for pred in outputs]
        print(outputs)
        assert len(outputs) == 1, outputs
        outputs = outputs[0]
        return outputs


    def _prep(self, topic: str, docs: List[str]):
        docs = list(map(lambda p: p.replace('  ', ' ').strip(), docs))
        docs = list(filter(bool, docs))
        # use the </s> token
        # TODO ... this needs better parameterization
        # 16k is not a good length choice
        tokenized_parts = [self.tokenizer(doc, add_special_tokens=True, truncation=True, max_length=16000 // len(docs)) for doc in docs]
        input_ids = []
        attention_mask = []
        global_attention_mask = []
        for tokenized_part in tokenized_parts:
            input_ids.extend(tokenized_part['input_ids'])
            #input_ids.append(docsep_id)
            attention_mask.extend(tokenized_part['attention_mask'])
            #attention_mask.append(1)
            global_attention_mask.extend([0] * (len(tokenized_part['input_ids'])))
            global_attention_mask[-1] = 1
            #global_attention_mask.append(1)

        #global_attention_mask[-1] = 0
        input_ids = input_ids[:16000 - 1]
        attention_mask = attention_mask[:16000 - 1]
        global_attention_mask = global_attention_mask[:16000 - 1]

        return torch.LongTensor([input_ids], device=self.model.device), torch.LongTensor([attention_mask], device=self.model.device)

    def supports_gpu(self) -> bool:
        return True

    def to(self, device: str):
        return self.model.to(device)
