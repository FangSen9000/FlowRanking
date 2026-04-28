## Takeaways

This project produced three concrete outcomes.

First, the training pipeline is now organized around direct comparison rather than single-model experimentation. The website, paper assets, result tables, and presentation materials all reflect the same benchmark plus the final `FlowRanking` system.

Second, the project established a realistic baseline picture. The classical methods are not weak strawmen: `ItemCF` is best on pointwise rating error and `BiasedMF` is best on ranking quality. `FlowRanking` uses this evidence directly and reaches the best value on all six required metrics.

Third, earlier neural ablations remain reproducible in the code, but they are no longer shown in the main public plots. This keeps the presentation focused on the deployed `FlowRanking` system and avoids confusing the final model with intermediate experiments.

Relevant project files are bundled directly on this page:

- paper PDF;
- presentation slides;
- code bundle;
- paper source bundle;
- raw final result text.
