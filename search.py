"""The VecturaDB part of the app: one collection with dense embeddings and a BM25 index, searched three ways."""
from pathlib import Path

import chromadb
from chromadb import K, Knn, Rrf, Schema, Search, SparseVectorIndexConfig
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction, DefaultEmbeddingFunction

HERE = Path(__file__).parent
# The dense embeddings: the default embedding function, a small local model (all-MiniLM-L6-v2), no API key. The
# collection uses it for the passages; here it embeds the questions.
embed = DefaultEmbeddingFunction()


def passages(folder=HERE / "docs"):
    """Each paragraph of each file in docs/ is one record. The first line of a file is its title."""
    for path in sorted(folder.glob("*.md")):
        title, *paragraphs = path.read_text(encoding="utf-8").strip().split("\n\n")
        for n, paragraph in enumerate(paragraphs):
            yield f"{path.stem}-{n}", paragraph.strip(), {"title": title.lstrip("# ")}


def open_handbook(path=HERE / "data"):
    """Opens the collection in the folder data/, and fills it with the handbook the first time."""
    client = chromadb.PersistentClient(path=str(path))
    # Next to the dense embeddings, a BM25 index: every document also gets a sparse vector of its words, under the
    # key "bm25".
    schema = Schema().create_index(
        key="bm25",
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT, embedding_function=ChromaBm25EmbeddingFunction(), bm25=True
        ),
    )
    handbook = client.get_or_create_collection("handbook", schema=schema)
    if handbook.count() == 0:
        ids, documents, metadatas = zip(*passages())
        handbook.add(ids=list(ids), documents=list(documents), metadatas=list(metadatas))
    return handbook


def search(handbook, question, limit=3):
    """The same question three ways, in one call: by meaning, by words, and both fused with RRF."""
    # The question is embedded once, for the dense search and for the hybrid one.
    vector = embed([question])[0]
    # In RRF, default=1000 keeps a passage that only one of the two rankings found, as if it were far down the other.
    fused = [Knn(query=vector, return_rank=True, default=1000), Knn(query=question, key="bm25", return_rank=True, default=1000)]
    ranks = {
        "dense": Knn(query=vector),
        "bm25": Knn(query=question, key="bm25"),
        "hybrid": Rrf(fused),
    }
    searches = [Search().rank(rank).limit(limit).select(K.DOCUMENT, K.SCORE, "title") for rank in ranks.values()]
    return dict(zip(ranks, handbook.search(searches).rows()))
