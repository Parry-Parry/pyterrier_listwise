import pandas as pd
from ._util import RankedList, split, iter_windows
import logging 
import numpy as np
from numpy import concatenate as concat
from enum import Enum

logger = logging.getLogger(__name__)

def _inner_pivot(model, qid : str, query : str, ranking : RankedList):
        '''
        l : current left partition being scored
        r : current right partition being the remainder of the array
        c : current candidates
        b : current backfill
        p : current pivot
        '''
        if model.verbose: logger.info(f"Processing query {qid} with {len(ranking)} documents")
        l = ranking[:model.window_size]
        r = ranking[model.window_size:]

        kwargs = {
            'qid': qid,
            'query': query,
            'doc_text': l.doc_texts,
            'doc_idx': l.doc_idx,
            'start_idx': 0,
            'end_idx': len(l), # initial sort may be less than window size
            'window_len': len(l)
        }

        order = np.array(model.score(**kwargs))
        orig_idxs = np.arange(len(l))
        l.doc_idx[orig_idxs], l.doc_texts[orig_idxs],  = l.doc_idx[order], l.doc_texts[order]
        logger.info(f"Initial sort complete for query {qid}, len: {len(l)}")
        if len(l) < model.window_size: 
            if model.verbose: logger.info('Breaking out')
            return l, r, True # breakout as only single sort is required
        p = l[model.cutoff]
        c = l[:model.cutoff]
        b = l[model.cutoff+1:]
        sub_window_size = model.window_size - 1 # account for addition of p

        while len(c) < model.buffer and len(r) > 0:
            l, r = split(r, sub_window_size)
            l = p + l

            kwargs = {
                'qid': qid,
                'query': query,
                'doc_text': l.doc_texts.tolist(),
                'doc_idx': l.doc_idx.tolist(),
                'start_idx': 0,
                'end_idx': len(l),
                'window_len': len(l)
            }

            order = np.array(model.score(**kwargs))
            orig_idxs = np.arange(len(l))
            l[orig_idxs] = l[order]

            p_idx = np.where(l.doc_idx == p.doc_idx[0])[0][0] # find index of pivot id
            # add left of pivot to candidates and right of pivot to backfill
            c = c + l[:p_idx]
            b = b + l[p_idx+1:]
        
        # we have found no candidates better than p
        if len(c) == model.cutoff - 1: 
            top = c + p 
            bottom = b + r
            return top, bottom, True 
        # we have found candidates better than p

        # split c by budget b
        c, ac = split(c, model.buffer)
        ac = ac + p + b + r

        return c, ac, False
    
def pivot(model, query : str, query_results : pd.DataFrame):
    qid = query_results['qid'].iloc[0]
    query_results = query_results.sort_values('score', ascending=False)
    doc_idx = query_results['docno'].to_numpy()
    doc_texts = query_results['text'].to_numpy()

    indicator = False
    num_iters = 0
    c = RankedList(doc_texts, doc_idx)
    b = RankedList()

    while not indicator and num_iters < model.max_iters:
        num_iters += 1
        c, _b, indicator = _inner_pivot(model, qid, query, c)
        b = b + _b
    if num_iters == model.max_iters:
        logger.warn(f"WARNING: max_iters reached for query {qid}")

    out = c + b
    return out.doc_idx, out.doc_texts

def sliding_window(model, query : str, query_results : pd.DataFrame):
        qid = query_results['qid'].iloc[0]
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        ranking = RankedList(doc_texts, doc_idx)
        for start_idx, end_idx, window_len in iter_windows(len(query_results), model.window_size, model.stride):
            kwargs = {
            'qid': qid,
            'query': query,
            'doc_text': ranking[start_idx:end_idx].doc_texts.tolist(),
            'doc_idx': ranking[start_idx:end_idx].doc_idx.tolist(),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'window_len': window_len
            }
            order = np.array(model.score(**kwargs))
            new_idxs = start_idx + order
            orig_idxs = np.arange(start_idx, end_idx)
            ranking[orig_idxs] = ranking[new_idxs]
        return ranking.doc_idx, ranking.doc_texts
    
def single_window(model, query : str, query_results : pd.DataFrame):
    qid = query_results['qid'].iloc[0]
    query_results = query_results.sort_values('score', ascending=False)
    candidates = query_results.iloc[:model.window_size]
    rest = query_results.iloc[model.window_size:]
    doc_idx = candidates['docno'].to_numpy()
    doc_texts = candidates['text'].to_numpy()
    rest_idx = rest['docno'].to_numpy()
    rest_texts = rest['text'].to_numpy()
    
    kwargs = {
        'qid': qid,
        'query': query,
        'doc_text': doc_texts.tolist(),
        'doc_idx': doc_idx.tolist(),
        'start_idx': 0,
        'end_idx': len(doc_texts),
        'window_len': len(doc_texts)
    }
    order = np.array(model.score(**kwargs))
    orig_idxs = np.arange(0, len(doc_texts))
    doc_idx[orig_idxs] = doc_idx[order]
    doc_texts[orig_idxs] = doc_texts[order]
    model.log.queries.append(model.current_query)

    return concat([doc_idx, rest_idx]), concat([doc_texts, rest_texts])
    
# from https://github.com/ielab/llm-rankers/blob/main/llmrankers/setwise.py

def _heapify(model, query, ranking, n, i):
    # Find largest among root and children
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    li_comp = model.score(**{
        'query': query['query'].iloc[0],
        'doc_text': [ranking.doc_texts[i], ranking.doc_texts[l]],
        'start_idx': 0,
        'end_idx': 1,
        'window_len': 2
    })
    rl_comp = model.score(**{
        'query': query['query'].iloc[0],
        'doc_text': [ranking.doc_texts[r], ranking.doc_texts[largest]],
        'start_idx': 0,
        'end_idx': 1,
        'window_len': 2
    })
    if l < n and li_comp == 0: largest = l
    if r < n and rl_comp == 0: largest = r

    # If root is not largest, swap with largest and continue heapifying
    if largest != i:
        ranking[i], ranking[largest] = ranking[largest], ranking[i]
        model._heapify(query, ranking, n, largest)

def setwise(model, query : str, query_results : pd.DataFrame):
    query_results = query_results.sort_values('score', ascending=False)
    doc_idx = query_results['docno'].to_numpy()
    doc_texts = query_results['text'].to_numpy()
    ranking = RankedList(doc_texts, doc_idx)
    n = len(query_results)
    ranked = 0
    # Build max heap
    for i in range(n // 2, -1, -1):
        _heapify(model, query, ranking, n, i)
    for i in range(n - 1, 0, -1):
        # Swap
        ranking[i], ranking[0] = ranking[0], ranking[i]
        ranked += 1
        if ranked == model.k:
            break
        # Heapify root element
        _heapify(model, query, ranking, i, 0)     
    return ranking.doc_idx, ranking.doc_texts  

class Mode(Enum):
    single = 1
    sliding = 2
    setwise = 4