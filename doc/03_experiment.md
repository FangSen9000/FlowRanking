## Experiments

### Final metrics

The final comparison is summarized below.

| Model | MAE↓ | RMSE↓ | Precision@10↑ | Recall@10↑ | F-measure@10↑ | NDCG@10↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ItemCF | 0.6578 | 0.8678 | 0.0645 | 0.0801 | 0.0596 | 0.0998 |
| BiasedMF | 0.8090 | 1.0212 | 0.1152 | 0.0906 | 0.0799 | 0.1439 |
| ClassicNeuMF | 0.7197 | 0.9406 | 0.0908 | 0.0650 | 0.0595 | 0.1150 |
| FlowRanking | 0.6568 | 0.8658 | 0.1335 | 0.1159 | 0.0970 | 0.1847 |

### What the numbers say

- `ItemCF` gives the best `MAE` and `RMSE` among the standalone components.
- `BiasedMF` is the strongest standalone component on Top-10 ranking metrics.
- `FlowRanking` improves those task-specific strengths with validation-calibrated ItemCF for rating prediction and fused BiasedMF/ItemCF scores for Top-10 ranking.
- The result is intentionally system-oriented: the final recommender uses the strongest components and then applies lightweight calibration/fusion to exceed the standalone baselines.

### Visual comparisons

<img src="./static/images/movielens/rating_metrics_bar.png" alt="Bar chart for MAE and RMSE">

**Figure 1:** Rating prediction metrics. Lower is better for both `MAE` and `RMSE`.

<img src="./static/images/movielens/topn_metrics_bar.png" alt="Bar chart for ranking metrics">

**Figure 2:** Top-N recommendation metrics. Higher is better for `Precision@10`, `Recall@10`, `F-measure@10`, and `NDCG@10`.

<img src="./static/images/movielens/precision_at_10.png" alt="Precision at 10 by model">
<img src="./static/images/movielens/recall_at_10.png" alt="Recall at 10 by model">
<img src="./static/images/movielens/f_measure_at_10.png" alt="F-measure at 10 by model">
<img src="./static/images/movielens/ndcg_at_10.png" alt="NDCG at 10 by model">

**Figure 3:** Metric-wise breakdown. These plots show that the final `FlowRanking` system exceeds the standalone components on both rating prediction and Top-10 recommendation.

### Interpretation

The main conclusion is that `FlowRanking` is the best final system for the assigned evaluation protocol: it calibrates `ItemCF` for rating prediction and fuses `BiasedMF` with `ItemCF` for ranking. Earlier neural ablations remain available in the code, but the public plots remove them so the final system is not confused with intermediate experiments.
