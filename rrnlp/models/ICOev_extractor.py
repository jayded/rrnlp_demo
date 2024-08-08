import ast
import os
import sys 
import warnings
warnings.filterwarnings("ignore")
from typing import Type, Tuple, List


import numpy as np 

import torch 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import rrnlp
from rrnlp.models import encoder, get_device

#weights_path = rrnlp.models.weights_path

#doi = rrnlp.models.files_needed['ev_inf_classifier']['zenodo']

weights = None
tokenizer_path = None

def get_ico_ev_extractor(weights: str, base_tokenizer: str, device) -> 'ICOBot':
    return ICOBot(*get_ico_ev_extractor_components(weights, base_tokenizer, device))

def get_ico_ev_extractor_components(weights: str, base_tokenizer: str, device) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    device = get_device(device=device)
    #dtype = torch.float32
    #print('loading to', device)
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
    os.makedirs('./offload', exist_ok=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        weights,
        #'mistralai/Mistral-7B-Instruct-v0.2',
        #is_trainable=False,
        device_map='auto',
        offload_folder='./offload',
        low_cpu_mem_usage=True,
    )
    return model, tokenizer

class ICOBot:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def predict_for_ab(self, ab: dict) -> Tuple[str, float]:
        #input_text = ab['ti'] + '  ' + ab['ab']
        input_text = ab['ab']
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=False).input_ids.to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False, decoder_input_ids=None)
        res = []
        for (_, dev_row), output in zip([input_text], outputs):
            out = tokenizer.decode([input_text], skip_special_tokens=True)
            #print(out, dev_row.to_dict())
            try:
                decoded = ast.literal_eval(out)
                for production in decoded:
                    if len(production) != 5:
                        print(f'error parsing {production} {dev_row["pmid"]}')
                        continue
                    try:
                        label = production[-1].split('[LABEL]')[1].split('[OUT]')[0].strip()
                    except:
                        print(f'label parse error for {production}')
                        label = 'MISSING'
                    components = ["Intervention", "Outcome", "Comparator", "Evidence", "Sentence", 'Label']
                    res.append(dict(zip(components, production)))
            except Exception as e:
                print("Error in decoding: ", dev_row, e)
        return res
