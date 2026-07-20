# NOTES

## Approach

I explored multiple strategies to improve speaker similarity beyond the provided baseline.

### 1. Stock Voice Evaluation
- Evaluated every stock voice embedding.
- Selected the highest-scoring voice as a baseline.

### 2. Search-Based Optimization
- Modified `search.py` to perform iterative optimization.
- Used:
  - Multi-sentence fitness evaluation
  - Random perturbations
  - Annealed step size
  - Simulated annealing acceptance
- This approach converged but did not outperform the baseline.

### 3. Pairwise Blend Search
- Ranked all stock voices by similarity.
- Selected the top five voices.
- Exhaustively evaluated weighted blends of every voice pair.
- Searched 21 blend weights for each pair (210 total candidates).

## Final Result

Best blend:

- `zm_yunxia`: 75%
- `hm_omega`: 25%

Best similarity achieved:

- **0.6488**

This blend was saved as the submitted `voice.pt`.

## Limitations

Given the assessment time limit, I focused on searching within the provided stock voice embeddings and linear blends rather than training or fine-tuning a new embedding.
