# VISTA — Topic Discovery Pipeline

**VISTA** is a lightweight, reproducible pipeline for turning raw texts and tweets into a topic-labeled tree/forest.

This repository is used in support of a scientific paper and contains the full preprocessing pipeline:
- `vectorisation.py` – embeddings and representations
- `vista_clustering.py` – distance matrix + clustering
- `vista_topics.py` – topic labeling and filtering

---

## 🧰 Requirements

- Python 3.10+
- OpenAI API key (`OPENAI_API_KEY`)
- Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 📁 Input Format

### `texts.json`
```json
[
  ["Paragraph 1 of doc A", "Paragraph 2 of doc A"],
  ["Another single-paragraph document"]
]
```

### `tweets.json`
```json
[
  "This is a tweet",
  "Another tweet here"
]
```

---

## 🚀 Pipeline Usage

Create an output folder:
```bash
mkdir -p artifacts
```

### 1. Vectorisation
```bash
python src/vectorisation.py \
  --texts data/texts.json \
  --tweets data/tweets.json \
  --output artifacts/items.json
```

### 2. Clustering
```bash
python src/vista_clustering.py \
  --input artifacts/items.json \
  --output artifacts/trees.json \
  --threshold 0.70
```

### 3. Topic Labeling
```bash
python src/vista_topics.py \
  --input artifacts/trees.json \
  --output artifacts/topics.json
```

---

## ⚙️ Environment Variables (Optional)

| Variable                | Default              | Description                              |
|-------------------------|----------------------|------------------------------------------|
| `OPENAI_API_KEY`        | –                    | OpenAI API key                           |
| `VISTA_EMBED_MODEL`     | text-embedding-3-small | Embedding model                          |
| `VISTA_CHAT_MODEL`      | gpt-4o-mini          | Chat model for summaries/topics          |
| `VISTA_OPENAI_TIMEOUT_S`| 60                   | OpenAI timeout in seconds                |
| `VISTA_CONCURRENCY`     | 5                    | Max concurrent OpenAI requests           |

---

## 🧪 Outputs

### `items.json`
Combined JSON with embeddings and summaries.

### `trees.json`
Forest base: cluster trees and disconnected items.

### `topics.json`
Final tree with topics assigned.

---

## 🧊 Notes

- Summaries and topics are generated via GPT-based models.
- spaCy is used for noun and stopword filtering.
- Code is intentionally designed for reproducibility and modularity.

---

## 📄 License & Citation

Include your license and citation information here.

> Last updated: 2025-09-24
