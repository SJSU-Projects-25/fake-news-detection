# Fake News Detection (improvements)

This repository contains a small notebook and datasets used for experimenting with fake-news detection.

Improvements added:
- `scripts/prepare_dataset.py`: robust dataset preparation script that validates, cleans, labels (Fake=0, True=1), shuffles and writes `data/news_dataset.csv`.

Quick usage:

```bash
# from project root
python3 scripts/prepare_dataset.py --fake data/Fake.csv --true data/True.csv --out data/news_dataset.csv
```

From a notebook you can run:

```ipython
%run scripts/prepare_dataset.py
```

If you don't have pandas installed:

```bash
pip install pandas
```

