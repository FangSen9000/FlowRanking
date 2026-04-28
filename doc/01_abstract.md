## Abstract

This project builds `FlowRanking`, a MovieLens recommendation system for both explicit rating prediction and Top-10 recommendation. We compare three recommender components under a unified training and evaluation setup: `ItemCF`, `BiasedMF`, and `ClassicNeuMF`, then deploy calibrated and fused task-specific components as the final `FlowRanking` system.

The project tracks both rating prediction metrics and ranking metrics. We report `MAE` and `RMSE` for pointwise accuracy, together with `Precision@10`, `Recall@10`, `F-measure@10`, and `NDCG@10` for recommendation quality.

The final system result is direct: `FlowRanking` reaches the best value on all six required course metrics by using validation-calibrated `ItemCF` for rating prediction and fused `BiasedMF`/`ItemCF` scores for Top-10 recommendation. Earlier neural ablations remain in the code for reproducibility, but the public results focus on the deployed system to avoid confusing the final model with intermediate experiments.
