The fitness function averaged speaker similarity across three sentences using Resemblyzer embeddings.

The search used structured perturbations on random tensor rows with an annealed step size and simulated annealing acceptance.

The best search result reached approximately 0.626 similarity.

The baseline blend produced a higher similarity score of 0.6331, so it was selected as the final submission.

The search likely plateaued because the optimization landscape is highly non-convex and perturbing a subset of tensor rows was insufficient to consistently improve the pretrained voice representation.

Given the assessment time limit, the strongest-performing tensor was submitted.
