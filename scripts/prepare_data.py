# scripts/prepare_data.py
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path


def main():
    """
    Downloads datasets (20 Newsgroups and AG News) and pre-computes their embeddings
    using a Sentence Transformer model, saving the results locally for reproducible experiments.

    This script handles the entire data preparation pipeline, ensuring consistent
    input data for all subsequent benchmark runs.
    """
    print("--- Data Preparation Script ---")

    # Define model and data paths
    MODEL_NAME = "all-MiniLM-L6-v2"
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)

    # Load the embedding model
    print(f"Loading sentence-transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # --- Process 20 Newsgroups ---
    print("\nProcessing 20 Newsgroups dataset...")
    # --- THIS IS THE CORRECTED LINE ---
    # The 'SetFit/20_newsgroups' dataset uses the 'default' config, not 'all'.
    dataset_20news = load_dataset("SetFit/20_newsgroups", split="train")
    df_20news = dataset_20news.to_pandas()

    print("Computing embeddings for 20 Newsgroups (this may take a few minutes)...")
    embeddings_20news = model.encode(df_20news["text"].tolist(), show_progress_bar=True)

    # Save embeddings and labels
    np.save(DATA_DIR / "20newsgroups_embeddings.npy", embeddings_20news)
    # The 'SetFit' version uses 'label_text' for human-readable labels and 'label' for integer labels.
    df_20news[["text", "label"]].to_csv(
        DATA_DIR / "20newsgroups_labels.csv", index=False
    )
    print("20 Newsgroups data saved.")

    # --- Process AG News ---
    print("\nProcessing AG News dataset...")
    # Using the canonical 'ag_news' dataset is more standard.
    dataset_agnews = load_dataset("ag_news", split="train")
    df_agnews = dataset_agnews.to_pandas()

    print("Computing embeddings for AG News (this may take a few minutes)...")
    embeddings_agnews = model.encode(df_agnews["text"].tolist(), show_progress_bar=True)

    # Save embeddings and labels
    np.save(DATA_DIR / "agnews_embeddings.npy", embeddings_agnews)
    df_agnews[["text", "label"]].to_csv(DATA_DIR / "agnews_labels.csv", index=False)
    print("AG News data saved.")

    print("\n--- Data preparation complete! ---")


if __name__ == "__main__":
    main()
