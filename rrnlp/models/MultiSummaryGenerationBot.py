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

from transformers import AutoModelForCausalLM, AutoTokenizer

from rrnlp.models import get_device

#weights_path = rrnlp.models.weights_path
# TODO
weights = ''

def load_mds_summary_bot(
        weights: str=weights, #hf model string or path
        device='auto',
    ):
    device = get_device(device)
    model = AutoModelForCausalLM.from_pretrained(weights).to(device)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    return MDSSummaryBot(model, tokenizer)


class MDSSummaryBot:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def predict_for_topic(self, topic: str, pmids: List):
        pass
