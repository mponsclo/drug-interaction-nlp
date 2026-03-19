"""Utility functions for NER feature file parsing.

Provides helpers to read tab-separated feature files produced by the
feature extractor and group them by sentence for CRF training/classification.
"""

from itertools import groupby
from typing import List, Tuple


def parse_sentence_strings(
    sentence: List[str],
) -> Tuple[List[Tuple[str, str, str]], List[List[str]], List[str]]:
    """Parse stringified sentence lines into structured data.

    Args:
        sentence: Lines from the feature file belonging to one sentence.

    Returns:
        Tuple of (tokens, features, tags) where:
        - tokens: list of (word, offset_from, offset_to) tuples
        - features: list of feature vectors per token
        - tags: list of BIO tags per token
    """
    tags = []
    features = []
    tokens = []
    for token in sentence:
        split_data = token.rstrip("\n").split("\t")
        features.append(split_data[5:])
        tags.append(split_data[4])
        tokens.append((split_data[1], split_data[2], split_data[3]))
    return tokens, features, tags


def read_feature_file(
    filepath: str,
) -> Tuple[
    List[Tuple[str, List[Tuple[str, str, str]]]],
    List[List[List[str]]],
    List[List[str]],
]:
    """Read a feature file and return data grouped by sentence.

    Args:
        filepath: Path to the tab-separated feature file.

    Returns:
        Tuple of (tokens_by_sentence, features, tags) where each element
        is a list with one entry per sentence.
    """
    features = []
    tags = []
    tokens_by_sentence = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    sentences = groupby(lines, lambda line: line.split("\t")[0])

    for sid, sentence in sentences:
        s_tokens, s_features, s_tags = parse_sentence_strings(list(sentence))
        tokens_by_sentence.append((sid, s_tokens))
        features.append(s_features)
        tags.append(s_tags)

    return tokens_by_sentence, features, tags
