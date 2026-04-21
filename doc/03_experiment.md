## Experiments

### Final metrics

The final comparison is summarized below.

| Model | MAE↓ | RMSE↓ | Precision@10↑ | Recall@10↑ | F-measure@10↑ | NDCG@10↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ItemCF | 0.6578 | 0.8678 | 0.0003 | 0.0000 | 0.0001 | 0.0004 |
| BiasedMF | 0.8089 | 1.0211 | 0.1408 | 0.0649 | 0.0704 | 0.1614 |
| ClassicNeuMF | 0.7204 | 0.9423 | 0.1084 | 0.0476 | 0.0524 | 0.1204 |
| FlowNeuMF | 0.7205 | 0.9421 | 0.1164 | 0.0490 | 0.0545 | 0.1334 |

### What the numbers say

- `ItemCF` gives the best `MAE` and `RMSE` on this reduced setting, which is a useful reminder that simple neighborhood methods can remain very competitive when the data regime is modest.
- `BiasedMF` is the strongest baseline on ranking metrics overall.
- `FlowNeuMF` does **not** beat `BiasedMF`, but it does beat `ClassicNeuMF` across every reported top-N metric while preserving essentially the same error scale on ratings.
- The clearest improvement from the proposed method is `NDCG@10`, where `FlowNeuMF` rises from `0.1204` to `0.1334`.

### Visual comparisons

<img src="./static/images/movielens/rating_metrics_bar.png" alt="Bar chart for MAE and RMSE">

**Figure 1:** Rating prediction metrics. Lower is better for both `MAE` and `RMSE`.

<img src="./static/images/movielens/topn_metrics_bar.png" alt="Bar chart for ranking metrics">

**Figure 2:** Top-N recommendation metrics. Higher is better for `Precision@10`, `Recall@10`, `F-measure@10`, and `NDCG@10`.

<img src="./static/images/movielens/precision_at_10.png" alt="Precision at 10 by model">
<img src="./static/images/movielens/recall_at_10.png" alt="Recall at 10 by model">
<img src="./static/images/movielens/f_measure_at_10.png" alt="F-measure at 10 by model">
<img src="./static/images/movielens/ndcg_at_10.png" alt="NDCG at 10 by model">

**Figure 3:** Metric-wise breakdown. These plots make it easier to see where the proposed method gains over `ClassicNeuMF` even when it is not the strongest overall baseline.

### Interpretation

The main conclusion is not that the flow-based idea already wins every benchmark. The stronger conclusion is narrower and more defensible: under a controlled comparison against the same neural backbone, the flow-style training strategy helps ranking quality. For a course project, that is an acceptable and meaningful result because it validates the direction of the proposed innovation.
