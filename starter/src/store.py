"""Persistent pgvector exact cosine search, isolated by collection and strategy."""
import math
from typing import Any

from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int,
                 collection: str = "day6") -> None:
        if type(dimensions) is not int or not 1 <= dimensions <= 16000:
            raise ValueError("Invalid embedding dimensions.")
        self.dimensions = dimensions
        self.collection = collection
        self.connection = psycopg.connect(database_url, connect_timeout=10,
                                          autocommit=True, row_factory=dict_row)

    def create_schema(self) -> None:
        self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.connection)
        self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS day6_chunks (
                collection text NOT NULL,
                strategy text NOT NULL CHECK (strategy IN ('A', 'B', 'C')),
                chunk_id text NOT NULL,
                document_id text NOT NULL,
                source text NOT NULL,
                page integer NOT NULL,
                text text NOT NULL,
                metadata jsonb NOT NULL,
                embedding vector({self.dimensions}) NOT NULL,
                PRIMARY KEY (collection, strategy, chunk_id)
            )
        """)
        column = self.connection.execute("""
            SELECT format_type(atttypid, atttypmod) AS type
            FROM pg_attribute WHERE attrelid = 'day6_chunks'::regclass
            AND attname = 'embedding'
        """).fetchone()
        if column["type"] != f"vector({self.dimensions})":
            raise ValueError("Existing table uses a different embedding dimension.")

    def _vector(self, values: list[float]) -> Vector:
        if len(values) != self.dimensions or not all(math.isfinite(x) for x in values):
            raise ValueError("Wrong vector dimensions or non-finite values.")
        if not any(values):
            raise ValueError("A zero vector has undefined cosine similarity.")
        return Vector(values)

    @staticmethod
    def _strategy(strategy: str) -> None:
        if strategy not in {"A", "B", "C"}:
            raise ValueError("Strategy must be A, B or C.")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Atomically replace this collection/strategy, including obsolete chunks."""
        self._strategy(strategy)
        if len(chunks) != len(vectors) or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Chunk/vector count mismatch or duplicate chunk IDs.")
        rows = [(self.collection, strategy, c.chunk_id, c.document_id, c.source,
                 c.page, c.text, Jsonb(c.metadata), self._vector(v))
                for c, v in zip(chunks, vectors)]
        with self.connection.transaction():
            self.connection.execute("DELETE FROM day6_chunks WHERE collection=%s AND strategy=%s",
                                    (self.collection, strategy))
            with self.connection.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO day6_chunks
                    (collection, strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, rows)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self._strategy(strategy)
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        filters = {} if filters is None else filters
        if (not isinstance(filters, dict) or set(filters) - {"chapter", "language"}
                or ("chapter" in filters and type(filters["chapter"]) is not int)
                or ("language" in filters and not isinstance(filters["language"], str))):
            raise ValueError("Unsupported metadata filter or value type.")
        vector = self._vector(query_vector)
        return self.connection.execute("""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s) AS score
            FROM day6_chunks
            WHERE collection=%s AND strategy=%s AND metadata @> %s
            ORDER BY embedding <=> %s, chunk_id
            LIMIT %s
        """, (vector, self.collection, strategy, Jsonb(filters), vector, top_k)).fetchall()

    def count(self, strategy: str) -> int:
        self._strategy(strategy)
        return self.connection.execute(
            "SELECT count(*) AS n FROM day6_chunks WHERE collection=%s AND strategy=%s",
            (self.collection, strategy)).fetchone()["n"]

    def versions(self) -> dict:
        return self.connection.execute("""
            SELECT current_setting('server_version') AS postgres,
                   extversion AS pgvector FROM pg_extension WHERE extname='vector'
        """).fetchone()

    def close(self) -> None:
        self.connection.close()
