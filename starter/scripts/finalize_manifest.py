import csv
import hashlib
import json
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PDF = BASE / "starter/data/raw/thinkpython2.pdf"
MANIFEST = BASE / "starter/data/manifest.json"

with (BASE / "results/page_audit.csv").open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
if not rows or any(
    r["include"] not in {"0", "1"} or not r["reason"].strip()
    for r in rows
):
    raise ValueError("Complete include=0/1 and reason for every page.")
if len({int(r["page"]) for r in rows}) != len(rows):
    raise ValueError("Duplicate page IDs in audit.")

selected = {int(r["page"]) for r in rows if r["include"] == "1"}
if len(selected) < 30:
    raise ValueError(f"Only {len(selected)} accepted pages; need at least 30.")

records = [
    json.loads(line)
    for line in (BASE / "results/candidate_pages.jsonl")
    .read_text(encoding="utf-8").splitlines() if line.strip()
]
accepted = [p for p in records if p["page"] in selected]
if len(accepted) != len(selected):
    raise ValueError("Audit and extracted pages do not match.")

accessed = date.fromisoformat(input("Actual download date (YYYY-MM-DD): "))
manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
manifest.update(
    accessed_at=accessed.isoformat(),
    local_path=PDF.relative_to(BASE.parent).as_posix(),
    sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
    content_page_count=len(accepted),
    page_numbering="Original PDF physical pages, starting at 1",
    pages=[
        {"page": p["page"], "characters": len(p["text"]), **p["metadata"]}
        for p in accepted
    ],
)
MANIFEST.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Manifest updated. Accepted pages:", len(accepted))