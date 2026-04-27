## Method

### Training setup

The implementation uses a common MovieLens preprocessing and split strategy, then trains all methods under the same recommendation target so the comparison remains fair. The final reported run uses controlled early stopping at `5` epochs because longer runs lowered training loss but degraded ranking quality.

The four methods are:

- `ItemCF`: a memory-based item-item collaborative filtering baseline.
- `BiasedMF`: a classical matrix factorization model with bias terms.
- `ClassicNeuMF`: a standard neural collaborative filtering model combining MF-style interactions with an MLP tower.
- `FlowNeuMF`: the proposed variant, which keeps the `NeuMF` backbone but adds a flow-inspired regularization mechanism.

The final system, `FlowRanking`, uses the best validated component for each required task:

- rating prediction: `ItemCF`;
- Top-10 recommendation: `BiasedMF`.

### Why FlowNeuMF?

The motivation is that standard neural collaborative filtering is usually trained with a direct objective on the endpoint prediction, while the proposed method tries to regularize the path by which the model approaches that endpoint. In the current implementation, the flow idea acts as an optimization prior: it encourages smoother movement in representation space and a more structured fitting process for the user-item interaction function.

That means the project is not claiming a full generative flow model for recommendations. Instead, it asks a practical question that is appropriate for a class project:

> If we inject a flow-style fitting strategy into a classical neural recommender, where does it help under the same data split and neural training setup?

### Practical debugging work

One important part of the project was engineering rather than theory. Earlier training runs were slower than expected, so the pipeline was inspected for bottlenecks, baseline consistency, and comparable hyperparameters. The final setup therefore focuses on:

- smaller data for faster iteration;
- direct baseline parity across methods;
- controlled early stopping to avoid overfitting the ranking task;
- reproducible outputs exported to the paper and the project website.

<img src="./static/images/movielens/training_loss_curves.png" alt="Training loss curves for MovieLens models">

**Figure:** Training curves from the final training script. These curves were used to inspect whether the flow-style model converged more slowly, whether extra epochs were needed, and whether the regularization behaved stably enough for comparison.
