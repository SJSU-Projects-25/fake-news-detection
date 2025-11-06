"""
scripts/prepare_dataset.py

Robust dataset preparation script for the Fake/True CSV files in `data/`.
Produces `data/news_dataset.csv` with columns from source plus a `label` column
(where Fake=0, True=1), shuffled and cleaned.

Usage:
    python3 scripts/prepare_dataset.py --fake data/Fake.csv --true data/True.csv --out data/news_dataset.csv

The script validates input files and required columns and prints a short summary.
"""

from pathlib import Path
import argparse
import logging
import sys

# Minimal runtime check for pandas
try:
    import pandas as pd
except Exception as e:
    print("pandas is required but not installed. Install it with: pip install pandas")
    raise


def prepare(fake_fp: Path, true_fp: Path, out_fp: Path, shuffle: bool = True, random_state: int = 42):
    """Load, validate, label, merge, clean and save the datasets.

    Returns the prepared DataFrame.
    """
    if not fake_fp.exists():
        raise FileNotFoundError(f"Fake dataset not found: {fake_fp}")
    if not true_fp.exists():
        raise FileNotFoundError(f"True dataset not found: {true_fp}")

    fake = pd.read_csv(fake_fp)
    true = pd.read_csv(true_fp)

    # Basic column expectations (commonly present in these datasets)
    # Allow slightly different column names; we'll look for title/text-like columns.
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    title_candidates = ["title", "Title", "headline"]
    text_candidates = ["text", "Text", "article"]

    fake_title = find_col(fake, title_candidates)
    true_title = find_col(true, title_candidates)
    fake_text = find_col(fake, text_candidates)
    true_text = find_col(true, text_candidates)

    if not (fake_title or fake_text):
        raise ValueError(f"`{fake_fp}` missing title/text-like columns. Found: {fake.columns.tolist()}")
    if not (true_title or true_text):
        raise ValueError(f"`{true_fp}` missing title/text-like columns. Found: {true.columns.tolist()}")

    # Normalise column names to 'title' and 'text' if possible
    def normalise(df, title_col, text_col):
        df = df.copy()
        if title_col and title_col != 'title':
            df['title'] = df[title_col]
        elif 'title' not in df.columns:
            df['title'] = ""
        if text_col and text_col != 'text':
            df['text'] = df[text_col]
        elif 'text' not in df.columns:
            df['text'] = ""
        return df

    fake = normalise(fake, fake_title, fake_text)
    true = normalise(true, true_title, true_text)

    fake = fake.copy()
    true = true.copy()
    fake['label'] = 0
    true['label'] = 1

    df = pd.concat([fake, true], ignore_index=True, sort=False)

    # Drop rows that have neither title nor text
    df = df.dropna(subset=['title', 'text'], how='all').reset_index(drop=True)

    # Fill missing strings
    df['title'] = df['title'].fillna('').astype(str)
    df['text'] = df['text'].fillna('').astype(str)

    # Optional shuffle
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Ensure output dir exists
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)

    return df


def parse_args():
    p = argparse.ArgumentParser(description="Prepare fake/true news dataset into a single CSV.")
    p.add_argument('--fake', default='data/Fake.csv', help='Path to Fake.csv')
    p.add_argument('--true', default='data/True.csv', help='Path to True.csv')
    p.add_argument('--out', default='data/news_dataset.csv', help='Output CSV path')
    p.add_argument('--no-shuffle', dest='shuffle', action='store_false', help='Disable shuffling')
    p.add_argument('--random-state', type=int, default=42, help='Random seed for shuffling')
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()
    fake_fp = Path(args.fake)
    true_fp = Path(args.true)
    out_fp = Path(args.out)

    logging.info(f"Loading: fake={fake_fp}, true={true_fp}")
    df = prepare(fake_fp, true_fp, out_fp, shuffle=args.shuffle, random_state=args.random_state)

    logging.info(f"Saved merged dataset to: {out_fp}")
    logging.info(f"Total rows: {len(df)}")
    logging.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")


if __name__ == '__main__':
    main()

