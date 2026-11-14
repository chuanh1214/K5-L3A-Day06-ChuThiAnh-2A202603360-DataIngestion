"""Local multilingual embeddings; never substitute mock vectors on failure."""
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Use identical normalized SentenceTransformer encoding for query/document."""

    def __init__(self, model_name: str, revision: str | None = None,
                 cache_folder: Path | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.manual_seed(0)
        torch.set_num_threads(4)
        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(model_name, revision=revision, device="cpu",
                                         cache_folder=str(cache_folder) if cache_folder else None)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.max_tokens = int(self.model.max_seq_length)

    def token_count(self, text: str) -> int:
        return len(self.model.tokenizer(text, truncation=False,
                                        add_special_tokens=True)["input_ids"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() or self.token_count(text) > self.max_tokens for text in texts):
            raise ValueError("Empty text or token overflow; refusing silent truncation.")
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 1, convert_to_numpy=True).tolist()

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]
