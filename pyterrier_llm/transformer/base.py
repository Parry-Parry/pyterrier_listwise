from ..algorithms import Mode, setwise, sliding_window, single_window
from functools import partial
from abc import ABC, abstractmethod
import pyterrier as pt
if not pt.started(): pt.init()
import pandas as pd
from tqdm import tqdm
import torch

class LLMTransformer(pt.Transformer, ABC):
    """
    Base class for re-ranking using LLMs.
    
    Parameters:
    window_size : int
        The size of the window to consider.
    stride : int
        The stride of the window.
    buffer : int
        The buffer size for pivot based ranking.
    cutoff : int
        The cutoff for pivot based ranking.
    n_child : int
        The number of child nodes to consider in setwise ranking.
    mode : str
        The ranking algorithm to use. One of 'setwise', 'sliding', 'single'.
    max_iters : int
        The maximum number of iterations for pivot based ranking.
    depth : int
        The maximum number of documents to consider.
    verbose : bool
        Whether to display progress bars.
    """

    def __init__(self, 
                 window_size : int = 20, 
                 stride : int = 10, 
                 buffer : int = 20, 
                 cutoff : int = 10, 
                 n_child : int = 3,
                 mode='sliding', 
                 max_iters : int = 100,
                 depth : int = 100,
                 verbose : bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.window_size = window_size
        self.stride = stride
        self.buffer = buffer
        self.cutoff = cutoff - 1
        self.num_child = n_child
        self.max_iters = max_iters
        self.depth = depth
        self.verbose = verbose

        assert cutoff < window_size, "cutoff must be less than window_size"
        assert mode in Mode, f"mode must be one of {Mode}"
        mode = {
            'setwise': setwise,
            'sliding': sliding_window,
            'single': single_window
        }[mode]
        self.mode = partial(mode, self)

    @abstractmethod
    def score(self, *args, **kwargs):
        raise NotImplementedError
    
    def transform(self, inp : pd.DataFrame):
            res = {
                'qid': [],
                'query': [],
                'docno': [],
                'text': [],
                'rank': [],
            }
            progress = not self.verbose
            for (qid, query), query_results in tqdm(inp.groupby(['qid', 'query']), unit='q', disable=progress):
                query_results.sort_values('score', ascending=False, inplace=True)
                with torch.no_grad():
                    doc_idx, doc_texts = self.mode(query, query_results.iloc[:self.depth])
                res['qid'].extend([qid] * len(doc_idx))
                res['query'].extend([query] * len(doc_idx))
                res['docno'].extend(doc_idx)
                res['text'].extend(doc_texts)
                res['rank'].extend(list(range(len(doc_idx))))
            res = pd.DataFrame(res)
            res['score'] = -res['rank'].astype(float)
            return res