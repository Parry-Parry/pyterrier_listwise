import numpy as np
from numpy import concatenate as concat
from tqdm import tqdm

def iter_windows(n, window_size, stride, verbose : bool = False):
    # TODO: validate window_size and stride
    for start_idx in tqdm(range((n // stride) * stride, -1, -stride, disable=verbose), unit='window'):
        end_idx = start_idx + window_size
        if end_idx > n:
            end_idx = n
        window_len = end_idx - start_idx
        if start_idx == 0 or window_len > stride:
            yield start_idx, end_idx, window_len

def split(l, i):
    return l[:i], l[i:]

class RankedList(object):
    def __init__(self, doc_idx=None, doc_texts=None) -> None:
        self.doc_texts = doc_texts if doc_texts is not None else np.array([])
        self.doc_idx = doc_idx if doc_idx is not None else np.array([])
    
    def __len__(self):
        return len(self.doc_idx)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RankedList(self.doc_idx[key], self.doc_texts[key])
        elif isinstance(key, int):
            return RankedList(np.array([self.doc_idx[key]]), np.array([self.doc_texts[key]]))
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            return RankedList([self.doc_idx[i] for i in key], [self.doc_texts[i] for i in key])
        else:
            raise TypeError("Invalid key type. Please use int, slice, list, or numpy array.")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.doc_idx[key], self.doc_texts[key] = value.doc_idx[0], value.doc_texts[0]
        elif isinstance(key, slice):
            self.doc_idx[key], self.doc_texts[key] = value.doc_idx, value.doc_texts
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            if len(key) != len(value):
                raise ValueError("Assigning RankedList requires the same length as the key.")
            for i, idx in enumerate(key):
                self.doc_idx[idx], self.doc_texts[idx] = value.doc_idx[i], value.doc_texts[i]

    def __add__(self, other):
        print(type(other))
        if not isinstance(other, RankedList):
            raise TypeError("Unsupported operand type(s) for +: 'RankedList' and '{}'".format(type(other)))
        return RankedList(concat([self.doc_idx, other.doc_idx]), concat([self.doc_texts, other.doc_texts]))
      
    def __str__(self):
      return f"{self.doc_idx}, {self.doc_texts}"