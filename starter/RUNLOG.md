# RUNLOG

## Run 1
Start: Best stock voice ranking using blend.py
Top voice: af_nova
Baseline similarity: 0.6331

## Run 2
Modified search.py:
- Multi-sentence fitness
- Random row perturbation
- Annealed step size
- Simulated annealing acceptance

Iterations: ~70
Best similarity: 0.6260

Observation:
The search converged but did not outperform the baseline. Audio quality remained acceptable, but similarity score was lower than the baseline blend.

Final decision:
Submitted the baseline blend tensor because it achieved the highest similarity score (0.6331).

# RUNLOG

## Baseline
- Evaluated all stock voices against the reference speaker.

## Blend Search
- Ranked stock voices by similarity.
- Selected the top 5 voices.
- Performed pairwise weighted blend search over the top voices.
- Tested 210 blend candidates (10 voice pairs × 21 weight settings).
- Best blend found:
  - zm_yunxia: 0.75
  - hm_omega: 0.25
- Best similarity score: **0.6488**

## Output
- Saved final voice embedding as `voice.pt`.
