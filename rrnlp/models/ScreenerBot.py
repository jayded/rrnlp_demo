import os
import sys 
from typing import Type, Tuple, List

import numpy as np 

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from bio import Entrez

import rrnlp
from rrnlp.models import get_device

#weights_path = rrnlp.models.weights_path
# TODO
weights = ''




#class PubmedQueryGeneratorBot:

