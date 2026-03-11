# partB/data — Dataset for Logical-Shapelets Reproduction

## Source

The dataset is obtained from **scikit-learn** (Python library). We use one of the following:

- **Option A**: `sklearn.datasets.make_classification(n_samples=200, n_features=50, n_informative=10, n_redundant=5, n_classes=2, random_state=42)`  
  This generates 200 samples, each with 50 features. We interpret each **row as one time series** of length 50 (each feature = one time point).

- **Option B**: `sklearn.datasets.load_digits()`  
  We take a subset (e.g. classes 0 and 1) so that we have at least 100 samples. Each sample is an 8×8 image flattened to 64 values; we treat it as a **time series of length 64**.

No external data files are required; the dataset is loaded in the notebooks via `sklearn.datasets` and optionally saved here for reference (e.g. as CSV) if needed.

## How It Is Used

- In **task 2 1.ipynb**: The dataset is loaded, preprocessed (e.g. standard scaling if needed), and cast into a matrix of shape `(n_samples, n_timesteps)`.
- In **task 2 2.ipynb** and **task 3** notebooks: The same matrix is used as input to the shapelet discovery code. Each row is one time series; the algorithm extracts subsequences and computes z-normalized distances as in the Logical-Shapelets paper (KDD 2011).

## Limitations Compared to the Paper

The paper uses the **UCR Time Series Classification Repository** (e.g. GunPoint, ECG200, gesture/robotics data) with real sequential signals. Our sklearn-based data:

- Is synthetic or image-derived, not true temporal recordings.
- May have different noise and structure than UCR (e.g. no clear “motif” in the middle of the series).
- Is used to demonstrate the **method** (shapelet discovery, information gain, logical combination) on a toy setup; we do not expect to match the paper’s reported accuracies on UCR.
