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

weights = '/work/frink/deyoung.j/evidence_inference_seq2seq/flant5_large_somin_repro/wlabels/excess/checkpoint-3500'
tokenizer_path = '/work/frink/deyoung.j/evidence_inference_seq2seq/flant5_large_somin_repro/wlabels/excess/checkpoint-3500'

def get_ico_ev_extractor(weights: str, base_tokenizer: str, device) -> 'ICOBot':
    return ICOBot(*get_ico_ev_extractor_components(weights, base_tokenizer, device))

def get_ico_ev_extractor_components(weights: str, base_tokenizer: str, device) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    device = get_device(device=device)
    print(f'Loading Model from {weights}')
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
    os.makedirs('./offload', exist_ok=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        weights,
        device_map='auto',
        offload_folder='./offload',
        low_cpu_mem_usage=True,
    )
    return model, tokenizer

class ICOBot:
    def __init__(
        self,
        device,
    ):
        self.model, self.tokenizer = get_ico_ev_extractor_components(weights=weights, base_tokenizer=tokenizer_path, device=device)

    def predict_for_ab(self, ab: dict) -> Tuple[str, float]:
        #input_text = ab['ti'] + '  ' + ab['ab']
        input_text = ab['ab']
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=False).input_ids.to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False, decoder_input_ids=None)
        res = []
        for dev_row, output in zip([input_text], outputs):
            out = self.tokenizer.decode(output, skip_special_tokens=True)
            try:
                decoded = ast.literal_eval(out)
                if len(decoded) == 0:
                    print(f'No ICO/Evidence relations found for {dev_row}')
                for production in decoded:
                    print(f'{dev_row}: {production}')
                    if len(production) != 5:
                        print(f'potential error parsing {production} {dev_row}')
                        continue
                    try:
                        label = production[-1].split('[LABEL]')[1].split('[OUT]')[0].strip()
                    except:
                        print(f'label parse error for {production}')
                        label = None
                    components = ["Intervention", "Outcome", "Comparator", "Evidence", "Sentence", 'Label']
                    ico_tuplet = dict(zip(components, production))
                    ico_tuplet['label'] = label
                    ico_tuplet['all'] = decoded
                    ico_tuplet['production'] = out
                    res.append(ico_tuplet)
            except Exception as e:
                res.append({'production': out})
                print("Error in decoding: ", out)
                print("Error in decoding: ", e)
                print("Error in decoding: ", dev_row)
        print(f'Final result: {res}')
        return res

    def supports_gpu(self) -> bool:
        return True

    def to(self, device: str):
        self.model.to(device)
