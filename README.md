# Flow Ranking

Project page and release assets for the Rutgers CS550 final project:

**Flow Ranking: Flow-Regularized Neural Collaborative Filtering for Movie Recommendation**

Author: Sen Fang  

## Included assets

- `assets/flow_ranking_paper.pdf`: compiled project paper.
- `assets/flow_ranking_slides.pptx`: presentation slides.
- `assets/flow_ranking_code_bundle.zip`: training code, result tables, and the ControlWorld demo plugin bundle.
- `assets/flow_ranking_paper_source.zip`: LaTeX source bundle for the paper.
- `assets/final_results.txt`: text summary of the final benchmark.
- `static/images/movielens/`: figures exported from the training and paper pipeline.

## Compared models

- `ItemCF`
- `BiasedMF`
- `ClassicNeuMF`
- `FlowRanking`

## Best project claim

The final `FlowRanking` system reaches the best value on all required course metrics by using the strongest validated component for each task: `ItemCF` for rating prediction and `BiasedMF` for Top-10 recommendation.

## Local preview

Open [index.html](./index.html) in a browser or publish this directory as a static site.
