"""Conservative full-quote evidence metrics, not human grounding labels."""
from collections import defaultdict
from typing import Any
import unicodedata


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    golds = [question, *question.get("gold_alternatives", [])]
    ranked = []
    for rank, hit in enumerate(hits[:3], start=1):
        matched = any(hit["document_id"] == gold["gold_source_id"]
                      and hit["page"] == gold["gold_page"]
                      and normalize(gold["evidence_quote"]) in normalize(hit["text"])
                      for gold in golds)
        ranked.append({**hit, "rank": rank, "evidence_hit": matched})
    return {"id": question["id"], "split": question["split"],
            "question": question["question"], "expected_answer": question["expected_answer"],
            "gold_source_id": question["gold_source_id"], "gold_page": question["gold_page"],
            "evidence_quote": question["evidence_quote"],
            "gold_alternatives": question.get("gold_alternatives", []),
            "requested_filter": question.get("filter"), "hits": ranked,
            "hit1": bool(ranked and ranked[0]["evidence_hit"]),
            "hit3": any(hit["evidence_hit"] for hit in ranked)}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for trace in traces:
        groups[(trace["strategy"], trace["split"])].append(trace)
    result = []
    for (strategy, split), rows in sorted(groups.items()):
        n = len(rows)
        n1 = sum(row["hit1"] for row in rows)
        n3 = sum(row["hit3"] for row in rows)
        filtered = [row for row in rows if row["requested_filter"] is not None]
        compliant = sum(len(row["hits"]) == 3 and all(
            all(hit["metadata"].get(k) == v for k, v in row["requested_filter"].items())
            for hit in row["hits"]) for row in filtered)
        result.append({"strategy": strategy, "split": split, "questions": n,
                       "hit1_count": n1, "hit1": n1 / n,
                       "hit3_count": n3, "hit3": n3 / n,
                       "filter_questions": len(filtered),
                       "filter_compliant_count": compliant,
                       "filter_compliance": compliant / len(filtered) if filtered else None})
    return {"metric": "Full quote on one gold source/page; NFC + whitespace only",
            "human_equivalent_evidence_review": "not performed", "results": result}
