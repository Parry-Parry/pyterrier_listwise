from typing import List
from .base import LLMTransformer
from ..modelling.openai import GPTRanker
import torch
import numpy as np
import os

class RankGPT(LLMTransformer):
    def __init__(self, 
                 checkpoint : str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = GPTRanker(model=checkpoint, context_size=4096, key=os.getenv('OPENAI_API_KEY'))
    
    def score(self, query : str, doc_text : List[str], window_len : int, **kwargs):
        order = self.model(query=query, texts=doc_text.tolist(), num=window_len)
        return np.array(order)