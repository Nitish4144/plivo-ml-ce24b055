# RUNLOG


## Run 1 – Stock Voice Baseline

**Method**
- Evaluated every stock voice embedding provided with the starter code.
- Synthesized the evaluation sentence for each voice.
- Measured cosine similarity against the target speaker embedding.

**Result**
- Best stock voice: `af_nova`
- Best similarity: **0.6331**

---

## Run 2 – Search-Based Optimization

**Method**
Modified `search.py` to perform iterative optimization using:
- Multi-sentence fitness evaluation
- Random row perturbations
- Simulated annealing
- Annealed step size
- Acceptance of occasional worse candidates to escape local optima

**Result**
- Approximately 70 optimization iterations
- Best similarity: **0.6260**

**Observation**
The optimization converged but did not outperform the baseline stock voice. Audio quality remained acceptable, but speaker similarity was lower.

---

## Run 3 – Blend Search

**Method**
Implemented a weighted blend search (`blend.py`):

1. Ranked all stock voices by similarity.
2. Selected the top five voices.
3. Evaluated every pair of the top voices.
4. Tested 21 blend weights for each pair.
5. Compared every synthesized sample against the target embedding.

**Search Space**
- Top voices: 5
- Voice pairs: 10
- Blend weights per pair: 21
- Total candidates evaluated: **210**

**Best Blend**
- `zm_yunxia`: **0.75**
- `hm_omega`: **0.25**

**Best Similarity**
- **0.6488**

---

## Final Submission

The pairwise weighted blend search produced the highest speaker similarity achieved during the assessment.

**Submitted Voice**
- `voice.pt`

**Similarity Score**
- **0.6488**