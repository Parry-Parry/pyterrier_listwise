# PyTerrier LLM

A PyTerrier plugin for LLM-based re-rankers. 

## Installation

```
pip install git+https://github.com/Parry-Parry/pyterrier_llm.git
```

## Current Models

* RankGPT: RankGPT model by Sun et al.
* PairT5:  Pairwise / Setwise Ranker based on FLAN-T5
* FastChatListwise: Allows use of models such as rankZephyr or RankVicuna for list-wise ranking
* LiT5: Allows use of FiD based LiT5, requires pyterrier_t5