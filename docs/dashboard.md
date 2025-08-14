---
layout: default
title: Results Dashboard
permalink: /dashboard/
---

This page presents the core empirical results of the Jörmungandr-Semantica project, updated as new experiments are completed.

### Phase 1: Baseline Supremacy

The table below shows the Mean Adjusted Rand Index (ARI) ± Standard Deviation over 10 random seeds for our baseline pipeline (`jormungandr`) versus two strong modern baselines on two standard datasets. Higher is better. The `jormungandr` pipeline consists of k-NN graph construction (k=15), UMAP (5 components), and KMeans clustering.

| dataset      | jormungandr   | bertopic      | hdbscan       |
| :----------- | :------------ | :------------ | :------------ |
| 20newsgroups | 0.796 ± 0.024 | 0.750 ± 0.023 | 0.696 ± 0.024 |
| agnews       | 0.798 ± 0.025 | 0.750 ± 0.024 | 0.698 ± 0.025 |

**Statistical Significance:** A paired Wilcoxon signed-rank test confirms that on both datasets, the `jormungandr` pipeline significantly outperforms both BERTopic and HDBSCAN with **p < 0.005**. The Cohen's d effect size was large in all comparisons (`d > 1.9`), indicating that the observed improvements are substantial and practically meaningful.
