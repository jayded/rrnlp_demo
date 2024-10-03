import json
import os
import sys 
from typing import Dict, List, Tuple, Type

import numpy as np 

#import faiss
from pymilvus import MilvusClient
import torch 
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from rrnlp.models import get_device

from numerizer import numerize
from nltk.tokenize import sent_tokenize

# nltk.download("punkt") # Uncomment this line if you haven't downloaded the punkt tokenizer before

outcome_type_model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
binary_outcomes_model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
continuous_outcomes_model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'


def get_numerical_extractor_bot() -> 'NumericalExtractorBot':
    # since we know, for the moment, that these model types are shared, let's not load the same model multiple times.
    model_types = set([outcome_type_model, binary_outcomes_model, continuous_outcomes_model])
    model_set = {x: AutoModelForCausalLM.from_pretrained(x) for x in model_types}
    tokenizer_set = {x: AutoTokenizer.from_pretrained(x) for x in model_types}
    models = {
        'outcome_type': model_set[outcome_type_model],
        'binary_outcomes': model_set[binary_outcomes_model],
        'continuous_outcomes': model_set[continuous_outcomes_model],
    }
    tokenizers = {
        'outcome_type': tokenizer_set[outcome_type_model],
        'binary_outcomes': tokenizer_set[binary_outcomes_model],
        'continuous_outcomes': tokenizer_set[continuous_outcomes_model],
    }
    for k, v in tokenizers.items():
        v.name = k
    return NumericalExtractorBot(
        outcome_type_model=NumericalExtractorBot.OutcomeTypeExtractor(
            model=models['outcome_type'],
            tokenizer=tokenizers['outcome_type'],
            new_tokens = 15, # llama generates some extra garbage so we up the o.g. token counts a little
        ),
        binary_outcomes_model=NumericalExtractorBot.FindingsExtractor(
            model=models['binary_outcomes'],
            tokenizer=tokenizers['binary_outcomes'],
            new_tokens = 60, # llama generates some extra garbage so we up the o.g. token counts a little
        ),
        continuous_outcomes_model=NumericalExtractorBot.FindingsExtractor(
            model=models['continuous_outcomes'],
            tokenizer=tokenizers['continuous_outcomes'],
            new_tokens = 80, # llama generates some extra garbage so we up the o.g. token counts a little
        ),
    )


def _convert_character_to_string_outcome_type(outcome_type: str) -> str:
    """
    This method converts the outcome type from character to string

    :param outcome_type: outcome type as character

    :return outcome type as string
    """
    character_to_string_mapping = {"A": "binary", "B": "continuous", "C": "x"}  # x is used to represent unknown
    outcome_type = outcome_type.replace("The answer is ", "").replace(".", "").replace("(", "").replace(")",
                                                                                                        "")  # remove any parens, periods, and other known common, extra texts
    # remove any unnecessary text output by finding the first non-space character
    for char in outcome_type:
        if not char.isspace():
            outcome_type = char
            break
    try:
        string_outcome = character_to_string_mapping[outcome_type]
    except:
        string_outcome = "x"  # x is used to represent unknown
    return string_outcome


# TODO: handle chunking for long texts in FindingsExtractor or find a better way to handle longer documents
#class InputChunker:
#    def __init__(self, tokenizer) -> None:
#        self.tokenizer = tokenizer
#
#    def __contains_digits(self, string_to_check: str) -> bool:
#        """
#        Check if a string contains digits.
#
#        Args:
#        string_to_check: string
#
#        Returns:
#        boolean
#        """
#        return any(char.isdigit() for char in string_to_check)
#    
#    def __preprocess_markdown(self, md_string: str) -> str:
#        """
#        Preprocess the markdown (md) string by converting number word to digits.
#        Then, remove the non-numerical text from the markdown string.
#        Tables do not get removed.
#
#        Args:
#        md_string: string
#
#        Returns:
#        processed_string: string with number words converted to digits
#        """
#        # Convert number words to digits
#        try:
#            numerized_text = numerize(md_string)
#        except:
#            print(f"ERROR: failed to numerize. using the following text as is : {md_string}")
#            numerized_text = md_string
#        
#        # Remove non-numerical text from the markdown string
#        processed_string = ""
#        special_string = "::::" # special string to split the regular text and tables
#        split_text = numerized_text.split(special_string)
#        for partial_text in split_text:
#            if "table wrap" in partial_text:
#                new_line = special_string + partial_text + special_string
#                processed_string += new_line
#            else:
#                # get list of setences
#                sentences = sent_tokenize(partial_text)
#                new_lines = []
#                for sent in sentences:
#                    if self.__contains_digits(sent):
#                        new_lines.append(sent)
#                joined_content = " ".join(new_lines)
#                processed_string += joined_content
#
#        return processed_string
#
#    def count_tokens(self, text: str) -> int:
#        """
#        Count the number of tokens in the text.
#        
#        Args:
#        text: string
#        
#        Returns:
#        token_count: integer
#        """
#        encoded = self.tokenizer.encode_text(text)
#        encoded_length = len(encoded)
#        return encoded_length
#
#    def __chunk_md_string(self, md_string: str, max_tokens: int) -> List[str]: # could technically do this with pre-processing
#        """
#        chunk the markdown string into chunks of ~max_tokens tokens
#
#        Args:
#        md_string: list
#        max_tokens: integer
#
#        Returns:
#        final_chunks: list
#        """
#        final_chunks = []
#        current_chunk = ""
#        current_length = 0
#
#        special_string = "::::" # special string to split the regular text and tables
#        split_text = md_string.split(special_string)
#
#        for partial_text in split_text:
#            if "table wrap" in partial_text:
#                table = special_string + partial_text + special_string
#                table_token_count = self.count_tokens(table)
#                if current_length + table_token_count > max_tokens:
#                    if current_length > 0:  # Avoid adding empty chunks
#                        chunk_to_add = {
#                            "chunk": current_chunk,
#                            "token_size": current_length
#                        }
#                        final_chunks.append(chunk_to_add)
#                        current_chunk = ""  # Reset the current chunk
#                        current_length = 0  # Reset the current length
#                    # Start a new chunk with the current table
#                    current_chunk = current_chunk + " " + str(table)
#                    current_length += table_token_count
#                else:
#                    # If adding this table wouldn't exceed max_tokens, add it to the current chunk
#                    current_chunk = current_chunk + " " + str(table)
#                    current_length += table_token_count
#            else:
#                sentences = sent_tokenize(partial_text)
#                for sentence in sentences:
#                    sentence_token_count = self.count_tokens(sentence)
#                    if current_length + sentence_token_count > max_tokens:
#                        if current_length > 0:  # Avoid adding empty chunks
#                            chunk_to_add = {
#                                "chunk": current_chunk,
#                                "token_size": current_length
#                            }
#                            final_chunks.append(chunk_to_add)
#                            current_chunk = ""  # Reset the current chunk
#                            current_length = 0  # Reset the current length
#                        # Start a new chunk with the current chunk
#                        current_chunk = current_chunk + " " + str(sentence)
#                        current_length += sentence_token_count
#                    else:
#                        # If adding this cbunk wouldn't exceed max_tokens, add it to the current chunk
#                        current_chunk = current_chunk + " " + str(sentence)
#                        current_length += sentence_token_count
#
#        # After the loop, add the current_chunk if it's not empty
#        if current_length > 0:
#            chunk_to_add = {
#                "chunk": current_chunk,
#                "token_size": current_length
#            }
#            final_chunks.append(chunk_to_add)
#
#        return final_chunks
#    
#    def get_chunked_input(self, md_string: str, max_chunk_token_size: int) -> List[str]:
#        """
#        Split a text into chunks of ~max_num_tokens tokens
#        
#        Args:
#        md_string: string
#        max_chunk_token_size: integer
#        
#        Returns:
#        chunked_input: A list of text chunks
#        """
#        processed_md_string = self.__preprocess_markdown(md_string)
#        chunks_list = self.__chunk_md_string(processed_md_string, max_chunk_token_size)
#        return chunks_list

class NumericalExtractorBot:
    class OutcomeTypeExtractor:
        def __init__(
            self,
            model,
            tokenizer,
            prompt_template,
            new_tokens,
        ):
            self.model = model
            self.tokenizer = tokenizer
            self.prompt_template = prompt_template
            self.new_tokens = new_tokens

        def predict_for_ab(self, ti_ab=None, pt_inputs=None, outcome=None):
            assert pt_inputs is None != ti_ab is None, "Must provide exactly one of pt_inputs or ti_ab"
            if pt_inputs is not None:
                assert outcome is None, "Cannot specify outcome text and inputs"
                model_inputs = pt_inputs
            elif ti_ab is not None:
                has_results_section = 'results' in ti_ab.keys()
                assert 'ab' in ti_ab.keys()
                assert 'outcome' is not None != 'outcome' in ti_ab.keys(), "must specify either outcome in ti_ab or as text"
                outcome = outcome if outcome is not None else ti_ab['outcome']
                if has_results_section:
                    text = ti_ab['ab'] + '\n\n' + ti_ab['results']
                else:
                    text = ti_ab['ab']
                text = prompt_template.format(abstract_and_results=text, outcome = outcome)
                model_inputs = tokenizer.apply_chat_template(text)
            else:
                assert False, "Must provide exactly one of pt_inputs or ti_ab"
            output = self.model.generate_output(model_inputs, max_new_tokens=self.max_new_tokens)
            outcome = _convert_character_to_string_outcome_type(outcome_predicted)
            return outcome
        
        def device(self):
            return self.model.device

    class FindingsExtractor:
        def __init__(
            self,
            model,
            tokenizer,
            prompt_template,
            new_tokens,
        ):
            self.model = model
            self.tokenizer = tokenizer
            self.prompt_template = prompt_template
            self.new_tokens = new_tokens

        # TODO consider batching
        def predict_for_ab(self, ti_ab=None, pt_inputs=None, intervention=None, comparator=None, outcome=None):
            assert pt_inputs is None != ti_ab is None, "Must provide exactly one of pt_inputs or ti_ab"
            if pt_inputs is not None:
                model_inputs = pt_inputs
            elif ti_ab is not None:
                assert (intervention is None == comparator is None) and (comparator is None == outcome is None), "Must provide all of ICO elements or pt_inputs, must not provide both"
                has_results_section = 'results' in ti_ab.keys()
                assert 'ab' in ti_ab.keys()
                if has_results_section:
                    text = ti_ab['ab'] + '\n\n' + ti_ab['results']
                else:
                    text = ti_ab['ab']
                text = prompt_template.format(abstract_and_results=text, intervention=intervention, comparator=comparator, outcome=outcome)
                model_inputs = tokenizer.apply_chat_template(text)
            else:
                assert False, "Must provide exactly one of pt_inputs or ti_ab"
            output = self.model.generate_output(model_inputs, max_new_tokens=self.max_new_tokens)

            return output
        
        def device(self):
            return self.model.device

    def __init__(
        self,
        outcome_type_model,
        binary_outcomes_extractor,
        continuous_outcomes_extractor,
    ):
        self.outcome_type_model = outcome_type_model
        self.binary_outcomes_extractor = binary_outcomes_exractor
        self.continuous_outcomes_extractor = continuous_outcomes_exractor


    # TODO handle results sections
    def predict_for_ab(self, ab: dict, ico_tuples: List[Dict[str, str]]) -> Tuple[str, float]:
        #input_text = ab['ti'] + '  ' + ab['ab']
        # first outcome type
        ab = ab.clone()
        outcomes = set([ico_tuple['outcome'] for ico_tuple in ico_tuples])
        outcome_to_type = dict()
        for outcome in outcomes:
            outcome_type = self.outcome_type_model.predict_for_ab(ti_ab=ti_ab, outcome=outcome)
            outcome_to_type[outcome] = outcome_type

        res = []
        for ico_tuple in ico_tuples:
            intervention = ico_tuple['intervention']
            comparator = ico_tuple['comparator']
            outcome = ico_tuple['outcome']
            outcome_type = outcome_to_type[outcome]
            ico_tuple = dict(ico_tuple.items())
            ico_tuple['outcome_type'] = outcome_type
            # then: binary outcomes extraction
            if outcome_type == 'binary':
                binary_result = self.binary_outcomes_extractor.predict_for_ab(ti_ab, intervention=intervention, comparator=comparator, outcome=outcome)
                ico_tuple['binary_result'] = binary_result
            # or continuous outcomes extraction
            elif outcome_type == 'continous':
                continuous_result = self.continuous_outcomes_extractor.predict_for_ab(ti_ab, intervention=intervention, comparator=comparator, outcome=outcome)
                ico_tuple['continuous_result'] = continuous_result
            else:
                assert False, "impossible state!"
            res.append(ico_tuple)
        return res
