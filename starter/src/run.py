"""Run a fixed local embedding/pgvector experiment: dev first, locked test next."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from dotenv import load_dotenv

from scripts.validate_benchmark import validate
from .chunking import by_paragraph, fit_token_limit, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .models import DocumentPage
from .store import PgVectorStore


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def load_inputs() -> tuple[dict, list[dict], list[DocumentPage]]:
    manifest_path = BASE / "starter/data/manifest.json"
    corpus_path = BASE / "results/candidate_pages.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if digest(ROOT / manifest["local_path"]) != manifest["sha256"]:
        raise ValueError("PDF no longer matches the finalized manifest.")
    questions = json.loads((ROOT / "benchmark/questions.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    errors = validate(questions, manifest, records)
    if errors:
        raise ValueError("Benchmark validation failed: " + "; ".join(errors))
    accepted = {p["page"]: p for p in manifest["pages"]}
    records = [r for r in records if r["page"] in accepted]
    if (len(records) < 30 or len(records) != manifest["content_page_count"]
            or len({r["page"] for r in records}) != len(records)
            or {r["page"] for r in records} != set(accepted)):
        raise ValueError("Accepted corpus pages are missing or duplicated.")
    for record in records:
        if (record["document_id"] != manifest["source_id"] or not record["text"].strip()
                or len(record["text"]) != accepted[record["page"]]["characters"]
                or any(record["metadata"].get(k) != accepted[record["page"]].get(k)
                       for k in ("chapter", "chapter_title", "language"))):
            raise ValueError("Extracted text no longer matches accepted page metadata.")
    return manifest, questions, [DocumentPage(**r) for r in records]


def configuration(manifest: dict) -> dict:
    return {"model": MODEL, "model_revision": REVISION, "device": "cpu",
            "document_and_query_encoding": "same encode(); normalized; no prefix",
            "metric": "cosine", "search": "exact; SQL filter before ranking", "top_k": 3,
            "A": {"chunking": "fixed", "size": 500, "overlap": 50},
            "B": {"chunking": "paragraph_line", "max_size": 500, "overlap": 0},
            "C": {"chunks_and_vectors": "identical to B", "filter": "question input only"},
            "token_guard": "split overflow chunks for A/B/C; never silently truncate",
            "pdf_sha256": manifest["sha256"],
            "manifest_sha256": digest(BASE / "starter/data/manifest.json"),
            "corpus_text_sha256": digest(BASE / "results/candidate_pages.jsonl"),
            "questions_sha256": digest(ROOT / "benchmark/questions.json"),
            "pipeline_source_sha256": {name: digest(BASE / "src" / name) for name in (
                "chunking.py", "embeddings.py", "store.py", "evaluation.py", "run.py")},
            "accepted_pages": manifest["content_page_count"]}


def chunk_stats(chunks: list, embedder: MyEmbedder) -> dict:
    lengths = sorted(len(c.text) for c in chunks)
    tokens = [embedder.token_count(c.text) for c in chunks]
    return {"chunk_count": len(chunks), "total_characters": sum(lengths),
            "characters_min": min(lengths), "characters_mean": mean(lengths),
            "characters_median": median(lengths),
            "characters_p95": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
            "characters_max": max(lengths), "total_tokens_including_special": sum(tokens),
            "max_tokens_including_special": max(tokens),
            "token_guard_chunks": sum("token_guard" in c.metadata["chunking"] for c in chunks)}


def ingest(store: PgVectorStore, pages: list[DocumentPage], embedder: MyEmbedder,
           output: Path) -> dict:
    stats = {}
    cached_b = None
    for strategy in ("A", "B", "C"):
        started = perf_counter()
        if strategy == "C":
            chunks, vectors = cached_b
            chunk_seconds = embedding_seconds = 0.0
        else:
            raw = [chunk for page in pages for chunk in (
                fixed_size(page, 500, 50) if strategy == "A" else by_paragraph(page, 500))]
            chunks = fit_token_limit(raw, embedder.token_count, embedder.max_tokens)
            chunk_seconds = perf_counter() - started
            embedding_started = perf_counter()
            vectors = embedder.embed_documents([c.text for c in chunks])
            embedding_seconds = perf_counter() - embedding_started
            if strategy == "B":
                cached_b = (chunks, vectors)
        write_started = perf_counter()
        store.upsert(strategy, chunks, vectors)
        write_seconds = perf_counter() - write_started
        first_count = store.count(strategy)
        reload_started = perf_counter()
        store.upsert(strategy, chunks, vectors)
        if store.count(strategy) != first_count or first_count != len(chunks):
            raise RuntimeError("Re-ingestion changed the expected chunk count.")
        stats[strategy] = {**chunk_stats(chunks, embedder),
                           "chunk_seconds": chunk_seconds, "embedding_seconds": embedding_seconds,
                           "database_write_seconds": write_seconds,
                           "ingest_seconds": chunk_seconds + embedding_seconds + write_seconds,
                           "reingest_check_seconds": perf_counter() - reload_started,
                           "reingest_count_verified": True, "paid_api_cost_usd": 0.0,
                           "embedding_reused_from": "B" if strategy == "C" else None}
        write_jsonl(output / f"chunks_{strategy}.jsonl", [asdict(c) for c in chunks])
        print(f"Indexed {strategy}: {len(chunks)} chunks; re-ingestion count verified", flush=True)
    return stats


def retrieve(store: PgVectorStore, questions: list[dict], embedder: MyEmbedder,
             stage: str) -> list[dict]:
    traces = []
    for question in (q for q in questions if q["split"] == stage):
        started = perf_counter()
        vector = embedder.embed_query(question["question"])
        query_seconds = perf_counter() - started
        for strategy in ("A", "B", "C"):
            filters = question.get("filter") if strategy == "C" else None
            search_started = perf_counter()
            hits = store.search(strategy, vector, filters=filters)
            trace = evaluate_query(question, hits)
            trace.update(strategy=strategy, applied_filter=filters,
                         query_embedding_seconds=query_seconds,
                         search_seconds=perf_counter() - search_started)
            traces.append(trace)
        print(f"Retrieved {stage}: {question['id']}", flush=True)
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, default=BASE / "results/retrieval")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(output / "pipeline.log"), encoding="utf-8",
                        level=logging.INFO, force=True)
    try:
        _run(args.stage, output)
    except Exception:
        logging.exception("Pipeline stopped; no mock backend or embedding fallback was used.")
        raise


def _run(stage: str, output: Path) -> None:
    manifest, questions, pages = load_inputs()
    config = configuration(manifest)
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    collection = "thinkpython2_" + signature
    lock_path = output / "config_lock.json"
    if stage == "dev" and (output / "test_summary.json").exists():
        raise ValueError("Test already evaluated; use a new output folder for another experiment.")
    if stage == "test":
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        selection = json.loads((output / "dev_selection.json").read_text(encoding="utf-8"))
        if lock["config"] != config or selection["collection"] != collection:
            raise ValueError("Inputs/config changed after dev; test is locked.")
        if (output / "test_summary.json").exists():
            raise ValueError("Test already evaluated; results will not be silently overwritten.")
    else:
        lock = {"locked_at": datetime.now(timezone.utc).isoformat(),
                "collection": collection, "config": config}
        write_json(lock_path, lock)
    load_dotenv(BASE / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing from starter/.env.")
    os.environ.setdefault("HF_HOME", str(BASE / ".cache/huggingface"))
    started = perf_counter()
    embedder = MyEmbedder(MODEL, revision=REVISION, cache_folder=BASE / "models")
    model_seconds = perf_counter() - started
    print(f"Model ready: {embedder.dimensions} dimensions; {embedder.max_tokens} token limit", flush=True)
    store = PgVectorStore(database_url, embedder.dimensions, collection)
    try:
        store.create_schema()
        if stage == "dev":
            stats = ingest(store, pages, embedder, output)
            write_json(output / "index_stats.json", stats)
        else:
            stats = json.loads((output / "index_stats.json").read_text(encoding="utf-8"))
        expected_counts = {s: stats[s]["chunk_count"] for s in ("A", "B", "C")}
        if any(store.count(s) != n for s, n in expected_counts.items()):
            raise RuntimeError("Persisted index counts differ from the recorded snapshot.")
        probe = embedder.embed_query(next(q["question"] for q in questions if q["split"] == "dev"))
        probe_ids = [h["chunk_id"] for h in store.search("B", probe)]
        store.close()
        store = PgVectorStore(database_url, embedder.dimensions, collection)
        store.create_schema()
        if (any(store.count(s) != n for s, n in expected_counts.items())
                or probe_ids != [h["chunk_id"] for h in store.search("B", probe)]):
            raise RuntimeError("Reopened database did not preserve counts and search results.")
        environment = {"timestamp": datetime.now(timezone.utc).isoformat(),
                       "model": MODEL, "revision": REVISION, "dimensions": embedder.dimensions,
                       "max_tokens": embedder.max_tokens, "model_load_seconds": model_seconds,
                       "collection": collection, "database": store.versions(),
                       "persistence_reopen_verified": True, "paid_api_cost_usd": 0.0,
                       "packages": {p: version(p) for p in (
                           "sentence-transformers", "transformers", "torch", "numpy", "pgvector", "psycopg")}}
        write_json(output / f"{stage}_environment.json", environment)
        traces = retrieve(store, questions, embedder, stage)
        write_jsonl(output / f"{stage}_traces.jsonl", traces)
        summary = summarize(traces)
        write_json(output / f"{stage}_summary.json", summary)
        if stage == "dev":
            ranked = sorted(summary["results"], key=lambda r: (
                -r["hit3"], -r["hit1"], stats[r["strategy"]]["chunk_count"], r["strategy"]))
            write_json(output / "dev_selection.json", {
                "selected_strategy": ranked[0]["strategy"], "collection": collection,
                "selected_before_test_at": datetime.now(timezone.utc).isoformat(),
                "criterion": "dev Hit@3; then Hit@1; then fewer chunks; then name for exact ties",
                "ranking": ranked})
        for row in summary["results"]:
            print(f"{row['strategy']} {stage}: Hit@1={row['hit1_count']}/{row['questions']}; "
                  f"Hit@3={row['hit3_count']}/{row['questions']}", flush=True)
        print(f"Saved outputs to {output}; no human audit labels were generated.", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
