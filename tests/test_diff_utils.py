"""Tests pour picarones.report.diff_utils."""

import pytest
from picarones.report.diff_utils import compute_word_diff, compute_char_diff, diff_stats


class TestComputeWordDiff:
    def test_equal_texts(self):
        ops = compute_word_diff("bonjour monde", "bonjour monde")
        assert len(ops) == 1
        assert ops[0]["op"] == "equal"
        assert ops[0]["text"] == "bonjour monde"

    def test_empty_reference(self):
        ops = compute_word_diff("", "bonjour")
        assert any(o["op"] == "insert" for o in ops)

    def test_empty_hypothesis(self):
        ops = compute_word_diff("bonjour", "")
        assert any(o["op"] == "delete" for o in ops)

    def test_both_empty(self):
        ops = compute_word_diff("", "")
        assert ops == []

    def test_insertion(self):
        ops = compute_word_diff("le chat", "le grand chat")
        assert any(o["op"] == "insert" and "grand" in o["text"] for o in ops)

    def test_deletion(self):
        ops = compute_word_diff("le grand chat", "le chat")
        assert any(o["op"] == "delete" and "grand" in o["text"] for o in ops)

    def test_replacement(self):
        ops = compute_word_diff("le chat dort", "le chien dort")
        assert any(o["op"] == "replace" and "chat" in o["old"] and "chien" in o["new"] for o in ops)

    def test_ops_cover_all_words(self):
        ref = "Bonjour monde médiéval"
        hyp = "Bonjour univers médiéval"
        ops = compute_word_diff(ref, hyp)
        # Reconstruction de la référence depuis les ops equal+delete+replace.old
        ref_reconstructed = []
        for op in ops:
            if op["op"] in ("equal", "delete"):
                ref_reconstructed.extend(op["text"].split())
            elif op["op"] == "replace":
                ref_reconstructed.extend(op["old"].split())
        assert ref_reconstructed == ref.split()

    def test_result_is_list_of_dicts(self):
        ops = compute_word_diff("texte", "text")
        assert isinstance(ops, list)
        assert all(isinstance(o, dict) for o in ops)

    def test_all_ops_have_op_key(self):
        ops = compute_word_diff("abc def ghi", "abc xyz ghi")
        assert all("op" in o for o in ops)

    def test_valid_op_types(self):
        valid_ops = {"equal", "insert", "delete", "replace"}
        ops = compute_word_diff("un deux trois", "un trois quatre")
        assert all(o["op"] in valid_ops for o in ops)


class TestComputeCharDiff:
    def test_equal(self):
        ops = compute_char_diff("abc", "abc")
        assert len(ops) == 1
        assert ops[0]["op"] == "equal"

    def test_single_char_replace(self):
        ops = compute_char_diff("abc", "axc")
        assert any(o["op"] == "replace" and o["old"] == "b" and o["new"] == "x" for o in ops)

    def test_empty_strings(self):
        assert compute_char_diff("", "") == []


class TestDiffStats:
    def test_empty(self):
        stats = diff_stats([])
        assert stats == {"equal": 0, "insert": 0, "delete": 0, "replace": 0}

    def test_counts(self):
        ops = [
            {"op": "equal", "text": "a"},
            {"op": "insert", "text": "b"},
            {"op": "delete", "text": "c"},
            {"op": "replace", "old": "d", "new": "e"},
            {"op": "equal", "text": "f"},
        ]
        stats = diff_stats(ops)
        assert stats["equal"] == 2
        assert stats["insert"] == 1
        assert stats["delete"] == 1
        assert stats["replace"] == 1
