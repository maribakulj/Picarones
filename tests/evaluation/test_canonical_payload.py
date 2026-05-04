"""Tests des helpers de :mod:`picarones.evaluation.projectors.canonical`.

Couvre les branches de :func:`canonical_payload_to_text` et
:func:`markdown_to_text` qui n'étaient pas exercées par les tests
des vues canoniques (S14/S16) — payloads dict/list, fallback ``str()``,
patterns markdown variés.
"""

from __future__ import annotations

from picarones.evaluation.projectors.canonical import (
    canonical_payload_to_text,
    markdown_to_text,
)


# ──────────────────────────────────────────────────────────────────
# markdown_to_text — patterns markdown courants
# ──────────────────────────────────────────────────────────────────


class TestMarkdownToText:
    def test_strips_headers(self) -> None:
        assert markdown_to_text("# Titre") == "Titre"
        assert markdown_to_text("## H2") == "H2"
        assert markdown_to_text("###### H6") == "H6"

    def test_strips_bullets(self) -> None:
        assert markdown_to_text("- élément") == "élément"
        assert markdown_to_text("* étoile") == "étoile"
        assert markdown_to_text("+ plus") == "plus"

    def test_strips_numbered_lists(self) -> None:
        assert markdown_to_text("1. premier") == "premier"
        assert markdown_to_text("42. quarante-deux") == "quarante-deux"

    def test_strips_blockquote(self) -> None:
        assert markdown_to_text("> citation") == "citation"
        assert markdown_to_text(">sans espace") == "sans espace"

    def test_strips_horizontal_rule(self) -> None:
        # Les HR sont supprimés.
        assert markdown_to_text("---").strip() == ""
        assert markdown_to_text("***") == ""

    def test_strips_bold_italic(self) -> None:
        assert markdown_to_text("**gras**") == "gras"
        assert markdown_to_text("*italique*") == "italique"
        assert markdown_to_text("***gras-italique***") == "gras-italique"

    def test_strips_underline(self) -> None:
        assert markdown_to_text("_souligné_") == "souligné"
        assert markdown_to_text("__double__") == "double"

    def test_strips_inline_code(self) -> None:
        assert markdown_to_text("`code`") == "code"

    def test_strips_code_blocks(self) -> None:
        text = "```python\nprint('hi')\n```"
        assert "print('hi')" in markdown_to_text(text)
        assert "```" not in markdown_to_text(text)

    def test_strips_links_keeps_text(self) -> None:
        assert markdown_to_text("[Picarones](https://example.com)") == "Picarones"

    def test_strips_images_keeps_alt(self) -> None:
        assert markdown_to_text("![alt](img.png)") == "alt"

    def test_combined(self) -> None:
        # Snippet réaliste VLM.
        md = "# Titre\n\n**Bonjour** _le_ `monde`\n\n- item 1\n- item 2"
        result = markdown_to_text(md)
        assert "Titre" in result
        assert "Bonjour" in result
        assert "monde" in result
        assert "item 1" in result
        # Pas de balise résiduelle.
        for marker in ("**", "##", "* ", "- ", "_", "`"):
            assert marker not in result.replace("- ", "")  # contre-faux-positif


# ──────────────────────────────────────────────────────────────────
# canonical_payload_to_text — dispatching par type
# ──────────────────────────────────────────────────────────────────


class TestCanonicalPayloadToText:
    def test_none_returns_empty(self) -> None:
        assert canonical_payload_to_text(None) == ""

    def test_str_treated_as_markdown(self) -> None:
        assert canonical_payload_to_text("# Titre\n\nBonjour") == "Titre\n\nBonjour"

    def test_int_falls_back_to_str(self) -> None:
        assert canonical_payload_to_text(42) == "42"

    def test_float_falls_back_to_str(self) -> None:
        assert canonical_payload_to_text(3.14) == "3.14"

    def test_dict_with_text_key(self) -> None:
        assert canonical_payload_to_text({"text": "Bonjour"}) == "Bonjour"

    def test_dict_with_content_key(self) -> None:
        assert canonical_payload_to_text({"content": "Hello"}) == "Hello"

    def test_dict_with_markdown_key(self) -> None:
        assert canonical_payload_to_text({"markdown": "# Titre"}) == "Titre"

    def test_dict_with_plain_key(self) -> None:
        assert canonical_payload_to_text({"plain": "brut"}) == "brut"

    def test_dict_with_value_key(self) -> None:
        assert canonical_payload_to_text({"value": "v"}) == "v"

    def test_dict_with_paragraphs_list(self) -> None:
        payload = {"paragraphs": ["para 1", "para 2", "para 3"]}
        result = canonical_payload_to_text(payload)
        assert "para 1" in result
        assert "para 2" in result
        assert "para 3" in result

    def test_dict_with_lines_list(self) -> None:
        payload = {"lines": ["ligne A", "ligne B"]}
        result = canonical_payload_to_text(payload)
        assert "ligne A" in result
        assert "ligne B" in result

    def test_dict_fallback_concatenates_string_values(self) -> None:
        # Aucune clé standard reconnue → on concatène les str du dict.
        payload = {"label1": "valeur 1", "label2": "valeur 2"}
        result = canonical_payload_to_text(payload)
        assert "valeur 1" in result
        assert "valeur 2" in result

    def test_dict_fallback_recurses_into_nested_dict(self) -> None:
        payload = {"nested": {"text": "inner"}}
        assert "inner" in canonical_payload_to_text(payload)

    def test_dict_fallback_recurses_into_nested_list(self) -> None:
        payload = {"items": ["a", "b"]}
        result = canonical_payload_to_text(payload)
        assert "a" in result
        assert "b" in result

    def test_list_concatenates_with_newlines(self) -> None:
        result = canonical_payload_to_text(["alpha", "beta", "gamma"])
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result

    def test_list_filters_empty_items(self) -> None:
        # Les éléments vides doivent être filtrés (pas de \n\n résiduel).
        result = canonical_payload_to_text(["alpha", "", "beta"])
        # Pas de double saut de ligne si on filtre bien les vides.
        assert "\n\n" not in result

    def test_tuple_treated_like_list(self) -> None:
        result = canonical_payload_to_text(("x", "y"))
        assert "x" in result
        assert "y" in result

    def test_list_of_dicts(self) -> None:
        payload = [{"text": "premier"}, {"text": "deuxième"}]
        result = canonical_payload_to_text(payload)
        assert "premier" in result
        assert "deuxième" in result

    def test_priority_text_over_content(self) -> None:
        # Les clés sont essayées dans l'ordre text > content > markdown.
        payload = {"text": "préféré", "content": "ignoré"}
        assert canonical_payload_to_text(payload) == "préféré"

    def test_non_str_value_in_known_key_skipped(self) -> None:
        # ``text`` doit être un str pour être pris ; sinon on continue
        # vers les clés suivantes ou le fallback.
        payload = {"text": 42, "content": "fallback"}
        assert canonical_payload_to_text(payload) == "fallback"
