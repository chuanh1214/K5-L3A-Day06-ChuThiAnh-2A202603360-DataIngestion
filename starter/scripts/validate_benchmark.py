"""Read-only validation of benchmark labels against the accepted local corpus."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "starter"
MANIFEST = STARTER / "starter/data/manifest.json"
PAGES = STARTER / "results/candidate_pages.jsonl"


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def validate(questions: object, manifest: dict, pages: list[dict]) -> list[str]:
    errors = []
    if not isinstance(questions, list) or len(questions) < 20:
        return ["Benchmark must be a list of at least 20 questions."]
    accepted = {p["page"] for p in manifest["pages"]}
    corpus = {(p["document_id"], p["page"]): p for p in pages}
    ids, texts = set(), set()
    counts = Counter()
    filter_counts = Counter()
    required = {
        "id", "split", "question", "expected_answer",
        "gold_source_id", "gold_page", "evidence_quote",
    }
    for index, question in enumerate(questions, start=1):
        label = f"Record {index}"
        if not isinstance(question, dict) or required - question.keys():
            errors.append(f"{label}: missing required fields or not an object.")
            continue
        qid = question["id"]
        if not isinstance(qid, str) or not qid.strip():
            errors.append(f"{label}: invalid ID.")
            continue
        if qid in ids:
            errors.append(f"{qid}: duplicate ID.")
        ids.add(qid)
        split = question["split"]
        if split not in ("dev", "test"):
            errors.append(f"{qid}: split must be dev or test.")
            continue
        counts[split] += 1
        if any(not isinstance(question[k], str) or not question[k].strip()
               for k in ("question", "expected_answer")):
            errors.append(f"{qid}: question and answer must be nonempty strings.")
            continue
        text = normalize(question["question"])
        if text.casefold() in texts:
            errors.append(f"{qid}: duplicate question text.")
        texts.add(text.casefold())
        filters = question.get("filter")
        valid_filter = filters is None or (
            isinstance(filters, dict) and set(filters) == {"chapter"}
            and type(filters["chapter"]) is int
        )
        if not valid_filter:
            errors.append(f"{qid}: filter must be null or an integer chapter filter.")
        elif filters is not None:
            filter_counts[split] += 1
            stated_chapters = re.findall(r"\bch\u01b0\u01a1ng\s+(\d+)\b", text.casefold())
            if str(filters["chapter"]) not in stated_chapters:
                errors.append(f"{qid}: filter chapter is not stated in the question.")
        alternatives = question.get("gold_alternatives", [])
        if not isinstance(alternatives, list):
            errors.append(f"{qid}: gold_alternatives must be a list.")
            alternatives = []
        for gold in [question, *alternatives]:
            if not isinstance(gold, dict):
                errors.append(f"{qid}: gold must be an object.")
                continue
            source, page = gold.get("gold_source_id"), gold.get("gold_page")
            quote = gold.get("evidence_quote")
            if source != manifest["source_id"] or type(page) is not int:
                errors.append(f"{qid}: invalid gold source or page type.")
                continue
            record = corpus.get((source, page))
            if page not in accepted or record is None:
                errors.append(f"{qid}: page {page} is missing or not accepted.")
                continue
            if not isinstance(quote, str) or not quote.strip():
                errors.append(f"{qid}: gold page {page} needs nonempty evidence.")
            elif normalize(quote) not in normalize(record["text"]):
                errors.append(f"{qid}: evidence does not occur on page {page}.")
            if valid_filter and filters is not None:
                if any(record["metadata"].get(k) != v for k, v in filters.items()):
                    errors.append(f"{qid}: filter excludes gold page {page}.")
    if counts["dev"] < 15 or counts["test"] < 5:
        errors.append("Need at least 15 dev and 5 test questions.")
    if sum(filter_counts.values()) < 3 or filter_counts["test"] < 1:
        errors.append("Need at least 3 filtered questions including 1 test question.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "benchmark/questions.json")
    args = parser.parse_args()
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
        pdf = ROOT / manifest["local_path"]
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("PDF SHA256 differs from the finalized manifest.")
        questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
        pages = [json.loads(line) for line in PAGES.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        errors = validate(questions, manifest, pages)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("Validation failed:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    counts = Counter(q["split"] for q in questions)
    filtered = [q["id"] for q in questions if q.get("filter") is not None]
    gold_count = sum(1 + len(q.get("gold_alternatives", [])) for q in questions)
    print(f"Benchmark OK: {len(questions)} questions; dev={counts['dev']}; test={counts['test']}")
    print(f"Filters: {', '.join(filtered)}; evidence entries verified: {gold_count}")
    print("Corpus SHA256 matches; all gold pages are accepted and satisfy their filters.")
    print("No retrieval was run. Answer semantics and final labels still need author review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
