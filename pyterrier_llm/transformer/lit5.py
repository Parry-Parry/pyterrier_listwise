from typing import List
import numpy as np
import torch
import re
from transformers import T5Tokenizer
from .base import LLMTransformer

class LiT5(LLMTransformer):
    template = "Search Query: {q} Passage: [{i}] {d} Relevance Ranking: "
    def __init__(self, 
                 model_path : str = 'castorini/LiT5-Distill-large', 
                 batch_size : int = 16, 
                 bfloat16 : bool = None, 
                 window_size : int = 20, 
                 **kwargs):
        from pyterrier_t5.modeling_fid import FiD
        super().__init__(**kwargs)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, return_dict=False, legacy=False, use_fast=True)
        self.model = FiD.from_pretrained(model_path, from_flax=False).cuda().eval()
        self.model.encoder.config.n_passages = window_size
        self.model.encoder.config.batch_size = batch_size
        if bfloat16 is None:
            try:
                self.model = self.model.bfloat16()
                bfloat16 = True
            except:
                bfloat16 = False
        elif bfloat16:
            self.model = self.model.bfloat16()
        self.bfloat16 = bfloat16

    def score(self, 
              query : str, 
              doc_text : List[str], 
              start_idx : int, 
              end_idx : int, 
              window_len : int, 
              **kwargs):
        passages = [self.template.format(q=query, i=i+1, d=text) for i, text in enumerate(doc_text.tolist() + ["" for _ in range(end_idx - start_idx, self.window_size)])]
        inputs = self.tokenizer.batch_encode_plus(passages, return_tensors="pt", padding='max_length', max_length=150, truncation=True)
        # get number of tokens in batch
        outputs = self.model.generate(
            input_ids=inputs['input_ids'].cuda().reshape(1, -1),
            attention_mask=inputs['attention_mask'].cuda().reshape(1, -1),
            max_length=100,
            do_sample=False,
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = re.sub(r'[^0-9]', ' ', output) # clean outputs (keep only digits)
        output = [int(x)-1 for x in output.split()] # convert to integer
        output = list({x: 0 for x in output if 0 <= x < window_len}.keys()) # remove duplicates (but keep order) and remove anything out of range
        order = output + [i for i in range(window_len) if i not in output] # backfill missing passages
        return np.array(order)