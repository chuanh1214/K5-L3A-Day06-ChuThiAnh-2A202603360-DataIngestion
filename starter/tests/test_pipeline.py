"""Focused boundary and metric tests; no model download or database required."""
import unittest

from src.chunking import by_paragraph, fit_token_limit, fixed_size
from src.evaluation import evaluate_query, summarize
from src.models import DocumentPage


class ChunkingTests(unittest.TestCase):
    def page(self, text):
        return DocumentPage("doc", "source.pdf", 7, text, {"chapter": 8})

    def check_chunks(self, page, chunks, limit):
        self.assertEqual(len(chunks), len({c.chunk_id for c in chunks}))
        covered = set()
        for chunk in chunks:
            start, end = chunk.metadata["start_char"], chunk.metadata["end_char"]
            self.assertEqual(chunk.text, page.text[start:end])
            self.assertLessEqual(len(chunk.text), limit)
            self.assertEqual((chunk.document_id, chunk.source, chunk.page), ("doc", "source.pdf", 7))
            self.assertEqual(chunk.metadata["chapter"], 8)
            covered.update(range(start, end))
        self.assertTrue(all(i in covered for i, c in enumerate(page.text) if not c.isspace()))

    def test_fixed_overlap_coverage_and_stable_ids(self):
        page = self.page("one two three four five six seven eight nine ten")
        chunks = fixed_size(page, 15, 4)
        self.check_chunks(page, chunks, 15)
        self.assertEqual(chunks, fixed_size(page, 15, 4))
        self.assertGreater(chunks[0].metadata["end_char"], chunks[1].metadata["start_char"])

    def test_paragraph_long_lines_heading_and_coverage(self):
        page = self.page("Intro line.\nMore context.\n\n8.2 New heading\n" + "x" * 83 + "\nend")
        self.check_chunks(page, by_paragraph(page, 24), 24)

    def test_token_guard_preserves_source_offsets(self):
        page = self.page("word " * 20 + "tail")
        chunks = fixed_size(page, 50, 5)
        guarded = fit_token_limit(chunks, lambda text: len(text) + 2, 19)
        self.check_chunks(page, guarded, 17)

    def test_invalid_parameters_and_empty_page(self):
        page = self.page("")
        self.assertEqual(fixed_size(page, 10, 2), [])
        self.assertEqual(by_paragraph(page, 10), [])
        for size, overlap in ((0, 0), (10, 10), (10, -1)):
            with self.assertRaises(ValueError):
                fixed_size(page, size, overlap)
        with self.assertRaises(ValueError):
            by_paragraph(page, 0)


class EvaluationTests(unittest.TestCase):
    def question(self):
        return {"id": "q", "split": "test", "question": "question", "expected_answer": "answer",
                "gold_source_id": "doc", "gold_page": 7, "evidence_quote": "enough evidence",
                "gold_alternatives": [{"gold_source_id": "doc", "gold_page": 8,
                                       "evidence_quote": "alternative evidence"}], "filter": None}

    def hit(self, text, page=7, document="doc"):
        return {"document_id": document, "page": page, "text": text, "metadata": {"chapter": 8}}

    def test_wrong_source_partial_quote_and_alternative(self):
        q = self.question()
        hits = [self.hit("enough evidence", document="other"),
                self.hit("enough"), self.hit("alternative\n evidence", page=8)]
        trace = evaluate_query(q, hits)
        self.assertFalse(trace["hit1"])
        self.assertTrue(trace["hit3"])
        self.assertFalse(evaluate_query(q, hits[:2])["hit3"])

    def test_evidence_split_between_chunks_is_not_a_hit(self):
        self.assertFalse(evaluate_query(self.question(),
                                      [self.hit("enough"), self.hit("evidence")])["hit3"])

    def test_empty_results_and_no_filter_are_not_perfect_scores(self):
        trace = evaluate_query(self.question(), [])
        trace["strategy"] = "A"
        result = summarize([trace])["results"][0]
        self.assertEqual(result["hit3_count"], 0)
        self.assertIsNone(result["filter_compliance"])


if __name__ == "__main__":
    unittest.main()
