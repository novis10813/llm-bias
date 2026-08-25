import importlib.util
from pathlib import Path


def _analysis_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "jspace_tfidf_analysis.py"
    spec = importlib.util.spec_from_file_location("jspace_tfidf_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ConceptNormalizer = _analysis_module().ConceptNormalizer


def test_concept_normalizer_consolidates_inflections() -> None:
    normalizer = ConceptNormalizer(set(), min_zipf=2.0)

    assert normalizer("drugs") == "drug"
    assert normalizer("tenants") == "tenant"
    assert normalizer("properties") == "property"
    assert normalizer("news") == "news"


def test_concept_normalizer_filters_token_pieces_but_keeps_domain_terms() -> None:
    normalizer = ConceptNormalizer({"noi", "specialterm"}, min_zipf=2.0)

    assert normalizer("renegot") is None
    assert normalizer("fint") is None
    assert normalizer("noi") == "noi"
    assert normalizer("specialterm") == "specialterm"
    assert normalizer("-store") == "store"
    assert normalizer("—including") == "include"
    assert normalizer("<|endoftext|>") is None
    assert normalizer("</think>") is None


def test_concept_normalizer_records_auditable_reasons() -> None:
    normalizer = ConceptNormalizer({"noi"}, min_zipf=2.0)

    normalizer("drugs")
    normalizer("noi")
    normalizer("renegot")

    assert normalizer.mapping["drugs"] == ("drug", "lemmatized")
    assert normalizer.mapping["noi"] == ("noi", "prompt_vocabulary")
    assert normalizer.mapping["renegot"] == (None, "filtered_low_frequency")
