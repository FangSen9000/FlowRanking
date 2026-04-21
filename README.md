# Flow Ranking

Project page and release assets for the Rutgers CS550 final project:

**Flow Ranking: Flow-Regularized Neural Collaborative Filtering for Movie Recommendation**

Author: Sen Fang  
Email: `sf895@scarletmail.rutgers.edu`

## Included assets

- `static/assets/flow_ranking_paper.pdf`: compiled project paper.
- `static/assets/flow_ranking_slides.pptx`: presentation slides.
- `static/assets/flow_ranking_code_bundle.zip`: code and result bundle for the recommender experiments.
- `static/assets/flow_ranking_paper_source.zip`: LaTeX source bundle for the paper.
- `static/assets/final_results.txt`: text summary of the final benchmark.
- `static/images/movielens/`: figures exported from the training and paper pipeline.

## Compared models

- `ItemCF`
- `BiasedMF`
- `ClassicNeuMF`
- `FlowNeuMF`

## Best project claim

The proposed `FlowNeuMF` model improves over `ClassicNeuMF` on all reported top-N ranking metrics while keeping nearly identical `RMSE`, which supports the flow-style training idea even though stronger classical baselines remain competitive.

## Local preview

Open [index.html](./index.html) in a browser or publish this directory as a static site.
