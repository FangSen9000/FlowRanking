## Method

### Training setup

The implementation uses a common MovieLens preprocessing and split strategy, then trains all methods under the same recommendation target so the comparison remains fair. The final reported run uses controlled early stopping at `5` epochs because longer runs lowered training loss but degraded ranking quality.

The public comparison uses:

- `ItemCF`: a memory-based item-item collaborative filtering baseline.
- `BiasedMF`: a classical matrix factorization model with bias terms.
- `ClassicNeuMF`: a standard neural collaborative filtering model combining MF-style interactions with an MLP tower.

The final system, `FlowRanking`, uses the best validated component for each required task:

- rating prediction: `ItemCF`;
- Top-10 recommendation: `BiasedMF`.

### Why FlowRanking?

The motivation is pragmatic: the required course tasks evaluate two different behaviors. Rating prediction rewards calibrated score estimates, while Top-10 recommendation rewards ranking relevant held-out items near the top. The final system therefore selects the component that performs best for each task instead of forcing one model to do both jobs.

> Which validated component should the final system use for each required recommendation task?

### Practical debugging work

One important part of the project was engineering rather than theory. Earlier training runs were slower than expected, so the pipeline was inspected for bottlenecks, baseline consistency, and comparable hyperparameters. The final setup therefore focuses on:

- smaller data for faster iteration;
- direct baseline parity across methods;
- controlled early stopping to avoid overfitting the ranking task;
- reproducible outputs exported to the paper and the project website.

<img src="./static/images/movielens/training_loss_curves.png" alt="Training loss curves for MovieLens models">

**Figure:** Training curves from the final training script. These curves were used to inspect whether the flow-style model converged more slowly, whether extra epochs were needed, and whether the regularization behaved stably enough for comparison.
