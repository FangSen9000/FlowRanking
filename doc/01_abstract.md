## Abstract

This project builds `FlowRanking`, a MovieLens recommendation system for both explicit rating prediction and Top-10 recommendation. We compare four recommender components under a unified training and evaluation setup: `ItemCF`, `BiasedMF`, `ClassicNeuMF`, and `FlowNeuMF`. The first two are strong classical baselines, while `FlowNeuMF` introduces a flow-style regularization strategy for the neural recommender.

The project tracks both rating prediction metrics and ranking metrics. We report `MAE` and `RMSE` for pointwise accuracy, together with `Precision@10`, `Recall@10`, `F-measure@10`, and `NDCG@10` for recommendation quality.

The final system result is direct: `FlowRanking` reaches the best value on all six required course metrics by using `ItemCF` for rating prediction and `BiasedMF` for Top-10 recommendation. The neural ablation is still informative: `FlowNeuMF` improves over `ClassicNeuMF` on rating error, showing that the flow-style regularization helps explicit rating prediction even though it is not the strongest standalone ranker on this dataset.
