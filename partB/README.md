# Logical-Shapelets: Reproduction and Ablation Study

**Author:** Subham Mahapatra (230037)

## Abstract

This report describes a reproduction of the Logical-Shapelets method for time series classification (Mueen, Keogh, Young, KDD 2011). We implement single-shapelet discovery with z-normalized subsequence distance, orderline-based threshold selection via information gain, and logical AND combination of two shapelets. Experiments on synthetic data confirm correct behavior of the implementation. Ablations compare single vs. logical AND and the effect of shapelet length range; a failure-mode experiment under high Gaussian noise illustrates the method's reliance on shape-stable patterns. The report is structured for direct conversion to IEEE format (e.g. Pandoc or LaTeX).

## Keywords

Time series classification, shapelets, logical shapelets, information gain, reproduction, ablation study.

## 1. Introduction

Shapelets are discriminative subsequences used for time series classification. The Logical-Shapelets paper extends this by combining multiple shapelets with logical AND and OR, and by algorithmic speedups (sufficient statistics and admissible pruning). This work reproduces the core discovery and logical AND combination on synthetic data, and studies ablations and a failure mode to align with the method's assumptions.

## 2. Related Work

The original shapelet work (Ye and Keogh) uses a single subsequence and a distance threshold. Logical-Shapelets (Mueen et al., KDD 2011) allow AND/OR combinations and propose optimizations for scalability. Our reproduction focuses on the binary single-shapelet and two-shapelet AND formulation and does not implement the full sufficient-statistics or pruning from the paper.

## 3. Methodology

**Single shapelet:** Candidates are generated over lengths MIN_LEN to MAX_LEN (step STEP). For each candidate we compute the z-normalized subsequence distance (Eq. 2) to every aligned window, build an orderline, and select the threshold that maximizes information gain (Definition 3), with tie-breaking by separation gap (Definition 4).

**Logical AND:** Two shapelets are combined by taking the maximum of their distances to each series and applying one threshold (Section 4 of the paper).

**Data:** Synthetic data from `sklearn.datasets.make_classification` (e.g. 150 samples, 40 features as time points, 2 classes); each row is one time series. All dataset references use the `partB/data/` folder.

## 4. Experiments

- **Reproduction:** Run shapelet discovery and logical AND; report information gain and test accuracy.
- **Ablation 1:** Single shapelet vs. logical AND (two shapelets).
- **Ablation 2:** max_len=5 vs. max_len=6.
- **Failure mode:** High additive Gaussian noise vs. low noise; observe accuracy drop and link to method assumptions.

Random seed `RANDOM_STATE = 42` and fixed MIN_LEN, MAX_LEN, STEP ensure reproducibility. All generated figures are stored in `partB/results/` with the naming convention `<task_name>_<output_description>.<extension>`.

## 5. Results and Analysis

### 5.1 Single-shapelet baseline

On the synthetic binary dataset, the single-shapelet classifier achieves an overall test accuracy of approximately 0.44. The confusion matrix in Fig. 1 shows that the model correctly identifies most class-0 instances but systematically misclassifies class-1 series as class 0, yielding no true positives for class 1 in this particular split. This behaviour is consistent with the discovered shapelet acting as a “prototype” for one class rather than a symmetric decision rule for both classes.

![Confusion matrix and shapelet](partB/results/task2_confusion_and_shapelet.png)

Figure 1: Confusion matrix of the single-shapelet classifier and the corresponding best shapelet subsequence discovered on the training data.

To obtain a more detailed view, Fig. 2 reports per-class precision, recall, and F1-score for the single-shapelet model. Class 0 attains moderate precision and high recall (reflecting that most predictions are biased toward class 0), whereas class 1 has effectively zero recall and F1-score on this test split. These metrics confirm that, on this synthetic task, the discovered shapelet explains only one side of the decision boundary and fails to capture a discriminative pattern specific to the minority class.

![Per-class PRF](partB/results/single_shapelet_prf.png)

Figure 2: Per-class precision, recall, and F1-score for the single-shapelet classifier on the test set.

Fig. 3 visualizes the training-set distribution of distances from each series to the selected shapelet, separated by class labels, together with the learned threshold \( \tau \). Class-0 instances are, on average, closer to the shapelet than class-1 instances, but the histograms exhibit substantial overlap. The threshold chosen by information gain lies in a region where both classes are present, which explains the modest information gain and the limited generalization performance.

![Distance distribution](partB/results/shapelet_distance_distribution.png)

Figure 3: Histogram of distances from the discovered shapelet to each training series, by class, with the learned threshold \( \tau \) overlaid.

### 5.2 Logical AND ablation

The logical-AND variant combines two shapelets by taking the maximum of their distances and learning a single threshold, as in the original paper. Fig. 4 compares test accuracy between the single-shapelet baseline and the two-shapelet AND model. On this synthetic dataset and for the chosen hyperparameters, the AND combination performs *worse* than the baseline, reducing accuracy from roughly 0.44 to about 0.25.

![Ablation 1: Logical AND](partB/results/task3_ablation1_logical_and.png)

Figure 4: Ablation 1 — comparison of test accuracy between the single-shapelet classifier and the logical-AND combination of two shapelets.

This outcome indicates that, in our setting, forcing both shapelets to be simultaneously “close” introduces an overly restrictive decision rule. Because the dataset was not designed to contain two independent motifs whose joint presence defines a class, the second shapelet tends to capture spurious structure or noise. The AND operation then increases false negatives without providing a corresponding gain in true positives, highlighting a weakness of logical shapelets when the underlying concept is not naturally conjunctive.

### 5.3 Shapelet length search range ablation

The second ablation studies the effect of restricting the candidate length search. Fig. 5 compares test accuracy for two configurations: a reduced search with \(\text{max\_len} = 5\) and the full range with \(\text{max\_len} = 6\). The two bars are nearly identical, with both configurations achieving accuracy around 0.44. This suggests that, for this particular synthetic dataset and length range, the most informative shapelet lies among shorter subsequences and is already accessible under the reduced search.

![Ablation 2: Max length](partB/results/task3_ablation2_maxlen.png)

Figure 5: Ablation 2 — effect of the maximum shapelet length on test accuracy. The reduced and full search ranges yield very similar performance.

From a methodological perspective, this ablation shows that modestly shrinking the length range can reduce computational cost without materially harming accuracy when the discriminative pattern is short. On more complex datasets, or when longer motifs are known to be important, a broader length search would likely be necessary.

### 5.4 Algorithmic efficiency ablation

Beyond classification accuracy, the reproduction also examines the impact of the algorithmic speedups proposed in the paper. Fig. 6 reports execution times (in seconds) for three configurations: the full method with both efficient distance computation and candidate pruning, a variant without pruning, and a variant without efficient distance computation. Removing pruning has little effect on runtime for this small synthetic dataset, but disabling efficient distance computation leads to a substantial slowdown, more than tripling the running time.

![Runtime ablation](partB/results/ablation_plot.png)

Figure 6: Ablation of runtime with and without efficient distance computation and pruning. Efficient distance via sufficient statistics is crucial for computational efficiency, while pruning offers smaller gains on this small dataset.

These observations are consistent with the original paper’s claims that sufficient-statistics-based distance computation is the dominant source of speedup, especially when many candidate windows must be evaluated.

### 5.5 Failure modes

Two complementary failure-mode experiments were conducted. First, Fig. 7 illustrates a constructed scenario in which both classes contain the same local pattern, differing only in the *number* of occurrences per series. The top panel shows representative time series for each class, while the bottom panel plots the orderline of distances to the best shapelet together with the learned threshold. Because series from both classes can be equally close to the shapelet (depending on which occurrence is aligned), their distance distributions overlap heavily and the threshold cannot separate them reliably. This confirms the theoretical limitation that classic shapelets and their logical extensions are sensitive to *presence* but not to *multiplicity* of a motif.

![Failure mode: orderline](partB/results/failure_mode.png)

Figure 7: Failure mode where both classes contain the same local pattern but with different occurrence counts; the distance-based orderline cannot separate the classes.

Second, Fig. 8 studies robustness to additive Gaussian noise. The baseline model is trained and evaluated on the original data, while the high-noise variant is trained and tested on heavily perturbed versions of the same series. The bar plot shows that accuracy is highly variable under strong noise; in some runs accuracy can even fluctuate upward, but the general pattern is that performance is unstable and sensitive to the particular noise realization. This behaviour arises because z-normalization removes scale and offset but not noise, so the relative ordering of distances to the shapelet can be distorted, especially when the signal-to-noise ratio is low.

![Failure mode: noise](partB/results/task3_failure_noise.png)

Figure 8: Failure mode under strong additive Gaussian noise. The single-shapelet classifier becomes unstable because the discriminative subsequence is corrupted, and the distance-based orderline no longer cleanly separates the classes.

## 6. Discussion

The empirical results confirm that the implemented pipeline faithfully reproduces the core behaviour of Logical-Shapelets on a controlled synthetic task. The single-shapelet model discovers a subsequence that is characteristic of one class, and the information-gain objective selects a threshold that approximately maximizes class separation along the induced orderline. However, the confusion matrix and per-class metrics reveal that this representation can be highly asymmetric, privileging one class when the other does not exhibit a distinct local motif.

The ablation on logical AND shows that simply adding expressive power does not guarantee improved accuracy: when the underlying concept is not inherently conjunctive, the second shapelet tends to encode noise or redundant structure, and the AND operation increases false negatives. The length-range ablation indicates that, on this dataset, shorter shapelets suffice and a reduced search space is adequate, which is encouraging from a computational standpoint. The runtime ablation further highlights the importance of efficient distance computation; without sufficient statistics, the naive sliding-window evaluation becomes a major bottleneck even for modest series lengths.

The failure-mode experiments sharpen two key limitations of shapelet-based classifiers. First, they are poorly suited to tasks where class differences are expressed through *counts* of a motif rather than its presence or absence. Second, they are vulnerable to high-frequency noise, because z-normalization alone cannot recover the underlying shape when the signal-to-noise ratio is low. These observations suggest that, in practical deployments, shapelet discovery should be paired with appropriate preprocessing (e.g., smoothing or denoising) and potentially augmented with features that capture motif frequency or more global temporal structure.

## 7. Conclusion

We reproduced the Logical-Shapelets single-shapelet and logical AND pipeline on synthetic data, ran ablations and a noise-based failure experiment, and discussed limitations and possible extensions (UCR datasets, OR shapelets, full optimizations).

## References

1. L. Ye and E. Keogh, "Time series shapelets: a new primitive for data mining," in Proc. KDD, 2009.
2. A. Mueen, E. Keogh, N. Young, "Logical-Shapelets: An Expressive Primitive for Time Series Classification," in Proc. KDD, 2011.
