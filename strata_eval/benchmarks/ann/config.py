"""Dataset definitions and default parameters for ANN benchmarks."""

# ANN benchmark datasets from ann-benchmarks.com (HDF5 format).
# Each entry specifies the download URL, vector dimension, distance metric,
# and expected sizes for the train (index) and test (query) splits.
ANN_DATASETS = {
    "sift-128-euclidean": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "dimension": 128,
        "metric": "euclidean",
        "train_size": 1000000,
        "test_size": 10000,
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "dimension": 100,
        "metric": "cosine",  # angular maps to cosine in Strata
        "train_size": 1183514,
        "test_size": 10000,
    },
    "glove-25-angular": {
        "url": "http://ann-benchmarks.com/glove-25-angular.hdf5",
        "dimension": 25,
        "metric": "cosine",
        "train_size": 1183514,
        "test_size": 10000,
    },
}

# Recall depths to evaluate (recall@1, recall@10, recall@100).
DEFAULT_K = [1, 10, 100]

# Number of vectors to upsert per batch during index build.
DEFAULT_BATCH_SIZE = 10000
