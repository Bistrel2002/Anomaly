"""Download the Credit Card Fraud Detection dataset from Kaggle.

Uses the ``kagglehub`` library to fetch the latest version of the
``mlg-ulb/creditcardfraud`` dataset and cache it locally inside a
``datasets/`` sub-directory next to this script.

Usage:
    python data/data.py
"""

import os

# Set a custom cache directory *before* importing kagglehub so the
# download lands inside the project tree rather than the user's home.
os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.path.dirname(__file__), "datasets")

import kagglehub  # noqa: E402  (must come after env-var override)


def main() -> None:
    """Download the dataset and print the local cache path."""
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()