## Method

### Training setup

The implementation uses a common MovieLens preprocessing and split strategy, then trains all methods under the same recommendation target so the comparison remains fair. The final reported run uses controlled early stopping at `5` epochs because longer runs lowered training loss but degraded ranking quality.

The public comparison uses:

- `ItemCF`: a memory-based item-item collaborative filtering baseline.
- `BiasedMF`: a classical matrix factorization model with bias terms.
- `ClassicNeuMF`: a standard neural collaborative filtering model combining MF-style interactions with an MLP tower.

The final system, `FlowRanking`, uses the best validated component for each required task and adds lightweight post-processing:

- rating prediction: validation-calibrated `ItemCF`;
- Top-10 recommendation: fused `BiasedMF`/`ItemCF` scores.

### Why FlowRanking?

The motivation is pragmatic: the required course tasks evaluate two different behaviors. Rating prediction rewards calibrated score estimates, while Top-10 recommendation rewards ranking relevant held-out items near the top. The final system therefore calibrates and fuses the strongest components instead of forcing one model to do both jobs.

> Which validated component should the final system use for each required recommendation task?

### Practical debugging work

One important part of the project was engineering rather than theory. Earlier training runs were slower than expected, so the pipeline was inspected for bottlenecks, baseline consistency, and comparable hyperparameters. The final setup therefore focuses on:

- smaller data for faster iteration;
- direct baseline parity across methods;
- controlled early stopping to avoid overfitting the ranking task;
- reproducible outputs exported to the paper and the project website.

<img src="./static/images/movielens/training_loss_curves.png" alt="FlowRanking precision-coverage and error CDF diagnostic curves">

**Figure:** Diagnostic curves for reliability and rating error. The precision-coverage curve shows how accurate high-confidence predictions remain as coverage grows, while the error CDF shows what fraction of predictions fall within each absolute-error threshold.
