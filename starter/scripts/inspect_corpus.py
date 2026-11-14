import csv
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from src.ingestion import load_pdf

BASE = Path(__file__).resolve().parents[1]
PDF = BASE / "starter/data/raw/thinkpython2.pdf"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)
(OUT / "pages").mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(OUT / "ingestion.log"),
    encoding="utf-8", level=logging.INFO, force=True,
)
pages = load_pdf(PDF, "thinkpython2-v2.4.1", str(PDF.relative_to(BASE)))
ranges = [
    (93, 104, 8, "Strings"),
    (105, 110, 9, "Case study: word play"),
    (111, 124, 10, "Lists"),
    (125, 136, 11, "Dictionaries"),
    (137, 146, 12, "Tuples"),
]
records, audit = [], []
for page in pages:
    for start, end, chapter, title in ranges:
        if start <= page.page <= end:
            page.metadata.update(
                chapter=chapter, chapter_title=title, language="en"
            )
            records.append(asdict(page))
            (OUT / "pages" / f"page_{page.page:03}.txt").write_text(
                page.text, encoding="utf-8"
            )
            audit.append([page.page, chapter, len(page.text), "", ""])
with (OUT / "page_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["page", "chapter", "characters", "include", "reason"])
    writer.writerows(audit)
with (OUT / "candidate_pages.jsonl").open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print("Physical pages:", len(pages))
print("Candidate pages:", len(records))
print("SHA256:", hashlib.sha256(PDF.read_bytes()).hexdigest())
print("Review outputs in:", OUT)