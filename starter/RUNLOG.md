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
