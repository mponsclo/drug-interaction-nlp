"""Utility functions for DDI feature file parsing.

Provides helpers to read tab-separated feature files produced by the
DDI feature extractor for MaxEnt training and classification.
"""

from typing import Dict, List, Tuple


def parse_string(
    line: str,
) -> Tuple[str, str, str, Dict[str, bool], str]:
    """Parse a single line from the feature file.

    Returns:
        Tuple of (sentence_id, e1_id, e2_id, features_dict, interaction_type).
    """
    split_data = line.rstrip("\n").split("\t")
    sentence_id = split_data[0]
    e1_id = split_data[1]
    e2_id = split_data[2]
    interaction = split_data[3]
    features = {f: True for f in split_data[4:]}
    return sentence_id, e1_id, e2_id, features, interaction


def read_feature_file(filepath: str) -> List[Tuple[Dict[str, bool], str]]:
    """Read a feature file for training (features + labels).

    Returns:
        List of (feature_dict, label) tuples for NLTK classifier training.
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if len(line.strip()) > 0:
                _, _, _, features, interaction = parse_string(line)
                data.append((features, interaction))
    return data


def read_test_feature_file(
    filepath: str,
) -> List[Tuple[str, str, str, Dict[str, bool]]]:
    """Read a feature file for testing (features without using labels).

    Returns:
        List of (sentence_id, e1_id, e2_id, feature_dict) tuples.
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if len(line.strip()) > 0:
                sentence_id, e1, e2, features, _ = parse_string(line)
                data.append((sentence_id, e1, e2, features))
    return data
