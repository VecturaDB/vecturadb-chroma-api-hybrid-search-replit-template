# Hybrid search with VecturaDB (Chroma API)

A small Python app that answers questions over the handbook of a fictional bike-share service, twenty short passages in `docs/`. It uses [VecturaDB](https://vecturadb.org), an open-source vector database compatible with the Chroma API, inside the app. Every question is searched three ways:

- **By meaning**: dense embeddings of a small local model, all-MiniLM-L6-v2, the model of the default embedding function. No API key.
- **By words**: a BM25 index. VecturaDB keeps a sparse vector of the words of each passage and scores the passages with BM25.
- **Hybrid**: the two rankings fused with Reciprocal Rank Fusion (RRF).

Try `E17`. The search by meaning puts the passage on code E23 first. BM25 finds the passage on E17, and the hybrid search keeps it first, with the closest passages after it.

## Run

The app runs at https://vecturadb-chroma-api-hybrid-search.replit.app.

To get your own copy on Replit, open https://replit.com/github.com/VecturaDB/vecturadb-chroma-api-hybrid-search-replit-template and press **Import from Github**. If Agent asks what to do with the project, choose **Leave it as imported**. Then press **Run**.

Elsewhere, with Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run main.py
```

Or with pip:

```bash
pip install VecturaDB==1.0.0rc5
python main.py
```

Then open http://127.0.0.1:5000. The three searches are also at `/api/search?q=<question>`, as JSON.

The first start downloads the embedding model, all-MiniLM-L6-v2 (Apache License 2.0, 83.2 MB), from where the default embedding function of `chromadb` downloads it, into `~/.cache/chroma/onnx_models`. Then it indexes the passages into `data/`. Later starts reuse both.

## How it works

The VecturaDB code is all in [`search.py`](search.py):

```python
embed = DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="data")
schema = Schema().create_index(
    key="bm25",
    config=SparseVectorIndexConfig(
        source_key=K.DOCUMENT, embedding_function=ChromaBm25EmbeddingFunction(), bm25=True
    ),
)
handbook = client.get_or_create_collection("handbook", schema=schema)

vector = embed([question])[0]
fused = [
    Knn(query=vector, return_rank=True, default=1000),
    Knn(query=question, key="bm25", return_rank=True, default=1000),
]
ranks = [Knn(query=vector), Knn(query=question, key="bm25"), Rrf(fused)]
searches = [Search().rank(rank).limit(3).select(K.DOCUMENT, K.SCORE, "title") for rank in ranks]
results = handbook.search(searches)
```

- `PersistentClient` runs VecturaDB in the Python process, with its data in `data/`. There is no server to start.
- The BM25 index is a sparse vector index on the documents: `add` computes the sparse vectors, and a text query is turned into one.
- The collection embeds the passages with the default embedding function, and `embed` embeds the questions with it. The question is embedded once, and its vector serves both the dense and the hybrid search.
- `default=1000` keeps in the hybrid ranking a passage that only one of the two searches found.
- In all three searches a lower score ranks first.

The `VecturaDB` package keeps the import name `chromadb`. It cannot be installed next to the `chromadb` package: see [Packages that depend on chromadb](https://github.com/VecturaDB/VecturaDB#packages-that-depend-on-chromadb).

## Your own documents

Put your files in `docs/`: the title on the first line, the passages separated by a blank line. Then delete `data/` and run the app again.

## Tested with

VecturaDB 1.0.0rc5 (release [1.0.0-preview.5](https://github.com/VecturaDB/VecturaDB/releases/tag/v1.0.0-preview.5)) and Python 3.12 on Linux x86_64, on 4 November 2026. Provided as is.

## Notice

Independent project. Not affiliated with, endorsed by or sponsored by Chroma Inc. or Replit, Inc. CHROMA is a registered trademark of Chroma Inc., used here only to describe compatibility. We use the name "Chroma" only to say which software this template works with. It does not mean that Chroma Inc. made, approved or supports it.

## License

Apache License 2.0: see [LICENSE](LICENSE).
