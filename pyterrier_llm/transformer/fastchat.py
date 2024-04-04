from typing import Optional, Union, List
import numpy as np
from .base import LLMTransformer
from ..modelling.prompt import RankPrompt
from ..modelling.fastchat import FastChatLLM
import torch

class FastChatListwise(LLMTransformer):
    def __init__(self, 
                 checkpoint : str, 
                 max_length : int = 300,
                 device : Union[str, int] = 'cuda', 
                 n_gpu : Optional[int] = 1, 
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = FastChatLLM(checkpoint=checkpoint, device=device, n_gpu=n_gpu)
        self.prompt = RankPrompt(model=checkpoint, tokenizer=self.model._tokenizer, max_length=max_length, rankllm=False)
    
    def score(self, query : str, doc_text : List[str], window_len : int, **kwargs):
        prompt = self.prompt(query=query, texts=doc_text.tolist(), num=window_len)
        order = self.model(text=prompt, window_len=window_len)
        return np.array(order)