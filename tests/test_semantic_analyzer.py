from model_architect.optimized_router import SemanticAnalyzer


def test_semantic_analyzer_extracts_enriched_features():
    analyzer = SemanticAnalyzer()
    features = analyzer.analyze(
        "Captain Mira Kade orders an immediate systems lockdown before sunrise."
    )

    assert -1.0 <= features.sentiment <= 1.0
    assert 0.0 <= features.complexity <= 1.0
    assert features.keywords, "Expected analyzer to surface topical keywords"
    # Entity extraction should gracefully degrade even without spaCy installed
    assert isinstance(features.entities, list)
