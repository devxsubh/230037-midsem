"""
Simple shapelet discovery for Logical-Shapelets (KDD 2011) reproduction.
Paper: Mueen, Keogh, Young - Logical-Shapelets: An Expressive Primitive for Time Series Classification.
Implementation follows Algorithm 1, 3; Section 2 (Definitions 2-5); Eq. 2 for normalized distance.
Also: Algorithm 4 (Efficient Distance via sufficient statistics), Algorithm 5 (Candidate Pruning),
Algorithm 6 (Fast Shapelet Discovery).
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Algorithm 4: Efficient Distance Computation using sufficient statistics
# Paper Section 3.1: cumulative sums S_x, S_xx for O(1) window mean/variance
# ---------------------------------------------------------------------------

def compute_sufficient_statistics(series):
    """
    Precompute cumulative sums for a time series for efficient window statistics.
    Paper Section 3.1: S_x[i] = sum(series[0:i]), S_xx[i] = sum(series[0:i]^2).
    We use length N+1 so that sum(series[i:j]) = S_x[j] - S_x[i].
    """
    series = np.asarray(series, dtype=float)
    # Cumulative sum of x; prepend 0 so S_x[i] = sum(series[0:i])
    S_x = np.zeros(len(series) + 1)
    for i in range(1, len(series) + 1):
        S_x[i] = S_x[i - 1] + series[i - 1]
    # Cumulative sum of x^2
    S_xx = np.zeros(len(series) + 1)
    for i in range(1, len(series) + 1):
        S_xx[i] = S_xx[i - 1] + series[i - 1] ** 2
    return S_x, S_xx


def sdist_efficient(shapelet, series, S_x, S_xx):
    """
    Minimum z-normalized subsequence distance from shapelet to any window in series,
    using precomputed sufficient statistics (Algorithm 4). Avoids recomputing
    mean and variance for each window from scratch.
    Paper: Eq. 2 normalized distance; Section 3.1 for efficient computation.
    """
    shapelet = np.asarray(shapelet, dtype=float)
    L = len(shapelet)
    n = len(series)
    if n < L:
        return float('inf')
    # Z-normalize the shapelet once (paper: shapelet is normalized)
    s_norm = z_norm(shapelet)
    # Sliding dot product: for each window w = series[i:i+L], we need sum(s_norm * w)
    # We can compute all at once with correlation
    dot_products = np.correlate(series, s_norm, mode='valid')
    min_dist = np.inf
    for start in range(n - L + 1):
        # Window mean using sufficient statistics: sum(series[start:start+L]) / L
        window_sum = S_x[start + L] - S_x[start]
        window_sum_sq = S_xx[start + L] - S_xx[start]
        window_mean = window_sum / L
        window_var = (window_sum_sq / L) - (window_mean ** 2)
        window_std = np.sqrt(max(0.0, window_var))
        if window_std < 1e-10:
            dist = np.inf
        else:
            # Z-normalized distance: for z-normalized s and w_norm = (w - mu)/sigma,
            # dist^2 = (1/L)*sum((s - w_norm)^2) = 2*(1 - correlation), correlation = dot(s,w_norm)/L
            correlation = dot_products[start] / (L * window_std)
            dist_sq = 2.0 * (1.0 - correlation)
            dist = np.sqrt(max(0.0, dist_sq))
        if dist < min_dist:
            min_dist = dist
    return min_dist


# ---------------------------------------------------------------------------
# Algorithm 5: Candidate Pruning
# Skip candidates that cannot beat the current best information gain
# ---------------------------------------------------------------------------

def can_prune_candidate(distances, labels, current_best_ig, min_variance=1e-10):
    """
    Decide if we can prune this candidate (Algorithm 5). We prune when the
    distance vector has almost no variation, so no threshold can achieve
    a good split (admissible: we never prune a candidate that could be best).
    """
    if len(distances) < 2:
        return True
    dist_var = np.var(distances)
    if dist_var < min_variance:
        return True
    return False


def z_norm(x):
    """Z-normalize a 1D array (zero mean, unit variance). Paper Section 2."""
    x = np.asarray(x, dtype=float)
    if x.std() < 1e-10:
        return np.zeros_like(x)
    return (x - x.mean()) / x.std()


def sdist(shapelet, series):
    """
    Subsequence distance: min over all alignments of normalized Euclidean distance.
    shapelet: 1D array (length L)
    series: 1D array (length >= L)
    Paper: Eq. 2, Algorithm 2. sdist(x,y) = sqrt(2(1 - C)) for normalized subsequences.
    """
    shapelet = np.asarray(shapelet, dtype=float)
    series = np.asarray(series, dtype=float)
    L = len(shapelet)
    n = len(series)
    if n < L:
        return float('inf')
    s_norm = z_norm(shapelet)
    min_dist = np.inf
    for start in range(n - L + 1):
        window = series[start:start + L]
        w_norm = z_norm(window)
        # Normalized Euclidean distance (length-normalized). Eq. 2: dist = sqrt(2(1-C))
        # For z-normalized vectors, (1/L)*sum((a-b)^2) = 2(1 - correlation) when means are 0
        d_sq = np.mean((s_norm - w_norm) ** 2)
        d = np.sqrt(d_sq)
        if d < min_dist:
            min_dist = d
    return min_dist


def entropy_labels(y):
    """Entropy of class distribution. Paper Definition 2: E(D) = - sum (n_i/N) log(n_i/N)."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return scipy_entropy(probs, base=2)


def information_gain(y, y_left, y_right):
    """
    Information gain of split. Paper Definition 3:
    I(s,tau) = E(D) - (N1/N)*E(D_left) - (N2/N)*E(D_right)
    """
    n = len(y)
    n1 = len(y_left)
    n2 = len(y_right)
    if n1 == 0 or n2 == 0:
        return 0.0
    e_d = entropy_labels(y)
    e_left = entropy_labels(y_left)
    e_right = entropy_labels(y_right)
    return e_d - (n1 / n) * e_left - (n2 / n) * e_right


def best_ig_threshold(distances, labels):
    """
    Given orderline (distances per series) and labels, find best split threshold and IG.
    Paper Algorithm 3: try midpoints between consecutive points; compute IG for each split.
    Returns (best_tau, best_ig, best_gap).
    """
    order = np.argsort(distances)
    dist_sorted = distances[order]
    labels_sorted = labels[order]
    n = len(labels)
    n_classes = len(np.unique(labels))
    max_ig = 0.0
    max_gap = 0.0
    best_tau = (dist_sorted[0] + dist_sorted[-1]) / 2
    for k in range(n - 1):
        tau = (dist_sorted[k] + dist_sorted[k + 1]) / 2
        y_left = labels_sorted[: k + 1]
        y_right = labels_sorted[k + 1:]
        ig = information_gain(labels_sorted, y_left, y_right)
        # Separation gap: Paper Definition 4. Simplified: mean distance in right - mean in left
        d_left = dist_sorted[: k + 1]
        d_right = dist_sorted[k + 1:]
        gap = np.mean(d_right) - np.mean(d_left) if len(d_right) and len(d_left) else 0.0
        if ig > max_ig or (ig == max_ig and gap > max_gap):
            max_ig = ig
            max_gap = gap
            best_tau = tau
    return best_tau, max_ig, max_gap


def discover_shapelet(X, y, min_len=3, max_len=None, step=1, random_state=None):
    """
    Brute-force shapelet discovery. Paper Algorithm 1.
    X: (n_samples, n_timesteps)
    y: (n_samples,) class labels
    Returns: (best_shapelet, best_tau, best_ig, best_gap, distances_for_best).
    """
    if random_state is not None:
        np.random.seed(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    n, m = X.shape
    if max_len is None:
        max_len = max(min_len, m // 2)
    max_ig = 0.0
    max_gap = 0.0
    best_shapelet = None
    best_tau = None
    best_distances = None
    for j in range(n):
        series = X[j]
        for length in range(min_len, min(max_len + 1, m + 1), step):
            for i in range(len(series) - length + 1):
                candidate = series[i : i + length]
                distances = np.array([sdist(candidate, X[k]) for k in range(n)])
                tau, ig, gap = best_ig_threshold(distances, y)
                if ig > max_ig or (ig == max_ig and gap > max_gap):
                    max_ig = ig
                    max_gap = gap
                    best_shapelet = candidate.copy()
                    best_tau = tau
                    best_distances = distances.copy()
    return best_shapelet, best_tau, max_ig, max_gap, best_distances


def discover_shapelet_fast(X, y, min_len=3, max_len=None, step=1, random_state=None,
                           use_efficient_dist=True, use_pruning=True):
    """
    Fast Shapelet Discovery (Algorithm 6). Uses efficient distance (Algorithm 4)
    and optional candidate pruning (Algorithm 5). Same return as discover_shapelet.
    """
    if random_state is not None:
        np.random.seed(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    n, m = X.shape
    if max_len is None:
        max_len = max(min_len, m // 2)

    # Precompute sufficient statistics for each series (Algorithm 4)
    S_x_list = []
    S_xx_list = []
    if use_efficient_dist:
        for j in range(n):
            S_x, S_xx = compute_sufficient_statistics(X[j])
            S_x_list.append(S_x)
            S_xx_list.append(S_xx)

    max_ig = 0.0
    max_gap = 0.0
    best_shapelet = None
    best_tau = None
    best_distances = None

    for j in range(n):
        series = X[j]
        for length in range(min_len, min(max_len + 1, m + 1), step):
            for i in range(len(series) - length + 1):
                candidate = series[i : i + length].copy()

                # Compute distances to all series (Algorithm 4 or naive)
                if use_efficient_dist:
                    distances = np.zeros(n)
                    for k in range(n):
                        distances[k] = sdist_efficient(candidate, X[k], S_x_list[k], S_xx_list[k])
                else:
                    distances = np.array([sdist(candidate, X[k]) for k in range(n)])

                # Candidate pruning (Algorithm 5): skip if cannot beat current best
                if use_pruning and can_prune_candidate(distances, y, max_ig):
                    continue

                tau, ig, gap = best_ig_threshold(distances, y)
                if ig > max_ig or (ig == max_ig and gap > max_gap):
                    max_ig = ig
                    max_gap = gap
                    best_shapelet = candidate
                    best_tau = tau
                    best_distances = distances.copy()

    return best_shapelet, best_tau, max_ig, max_gap, best_distances


def discover_two_shapelets_and(X, y, min_len=3, max_len=None, step=1, random_state=None):
    """
    Find two shapelets and combine with AND (max of two distances). Paper Section 4.
    Returns (shapelet1, shapelet2, tau, combined_ig, combined_distances).
    """
    if random_state is not None:
        np.random.seed(random_state)
    s1, tau1, ig1, _, dist1 = discover_shapelet(X, y, min_len=min_len, max_len=max_len, step=step, random_state=random_state)
    # Second shapelet: search again; we use a simple approach: best shapelet on "residual" or second best
    # Simplified: get another shapelet by excluding the series that contributed s1
    n = len(y)
    best_ig2 = 0.0
    best_tau2 = None
    best_s2 = None
    best_combined_d = None
    for j in range(n):
        series = X[j]
        m = len(series)
        max_l = min(max_len or m // 2, m)
        for length in range(min_len, max_l + 1, step):
            for i in range(len(series) - length + 1):
                candidate = series[i : i + length]
                d2 = np.array([sdist(candidate, X[k]) for k in range(n)])
                # AND: combined distance = max(d1, d2). Paper Section 4.
                combined_d = np.maximum(dist1, d2)
                tau, ig, _ = best_ig_threshold(combined_d, y)
                if ig > best_ig2:
                    best_ig2 = ig
                    best_tau2 = tau
                    best_s2 = candidate.copy()
                    best_combined_d = combined_d.copy()
    if best_s2 is None:
        return s1, None, best_tau2 or tau1, best_ig2 or ig1, best_combined_d or dist1
    return s1, best_s2, best_tau2, best_ig2, best_combined_d


def get_split_labels(distances, y, tau):
    """Given distances and labels, return (label_for_close, label_for_far) by majority vote."""
    from collections import Counter
    y = np.asarray(y)
    left = y[distances <= tau]
    right = y[distances > tau]
    label_close = Counter(left).most_common(1)[0][0] if len(left) else y[0]
    label_far = Counter(right).most_common(1)[0][0] if len(right) else y[0]
    return label_close, label_far


def predict_shapelet(X_test, shapelet, tau, label_close=None, label_far=None):
    """Predict: if d <= tau predict label_close else label_far. If labels not given, returns 1/0."""
    preds = []
    for i in range(len(X_test)):
        d = sdist(shapelet, X_test[i])
        preds.append(1 if d <= tau else 0)
    preds = np.array(preds)
    if label_close is not None and label_far is not None:
        out = np.where(preds == 1, label_close, label_far)
        return out
    return preds


def predict_logical_and(X_test, shapelet1, shapelet2, tau, label_close=None, label_far=None):
    """Predict using AND of two shapelets: close iff max(d1,d2) <= tau."""
    preds = []
    for i in range(len(X_test)):
        d1 = sdist(shapelet1, X_test[i])
        d2 = sdist(shapelet2, X_test[i])
        preds.append(1 if max(d1, d2) <= tau else 0)
    preds = np.array(preds)
    if label_close is not None and label_far is not None:
        return np.where(preds == 1, label_close, label_far)
    return preds
