"""Classify tokens using a trained CRF model for Named Entity Recognition.

Loads a pre-trained CRF model and applies it to extracted features,
outputting entity predictions in the evaluation format.
"""

import argparse
from typing import List, Tuple

import pycrfsuite

from utils import read_feature_file


def output_entities(
    sid: str, tokens: List[Tuple[str, str, str]], tags: List[str]
) -> None:
    """Print recognized entities in the format expected by the evaluator.

    Merges consecutive B-I tokens into multi-word entities.

    Args:
        sid: Sentence identifier.
        tokens: List of (word, offset_from, offset_to) tuples.
        tags: List of BIO tags for each token.
    """
    i = 0
    while i < len(tokens):
        entity, offset_from, offset_to = tokens[i]
        tag = tags[i]

        if tag[0] == "B":
            tag_name = tag[2:]
            j = i + 1
            while j < len(tokens):
                word_next, offset_from_next, offset_to_next = tokens[j]
                tag_next = tags[j]
                j += 1
                if int(offset_from_next) - int(offset_to) != 2 or tag_next[0] != "I":
                    break
                if tag_next[2:] == tag_name:
                    entity = entity + " " + word_next
                    offset_to = offset_to_next
            print(f"{sid}|{offset_from}-{offset_to}|{entity}|{tag_name}")

        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRF-based NER classifier.")
    parser.add_argument("model_name", type=str, help="Path to trained CRF model")
    parser.add_argument("dataset_path", type=str, help="Path to feature file")
    args = parser.parse_args()

    sentence_ids_and_tokens, X_devel, y_devel = read_feature_file(args.dataset_path)

    tagger = pycrfsuite.Tagger()
    tagger.open(args.model_name)
    tags = [tagger.tag(sentence) for sentence in X_devel]

    for sentence_tags, sentence_data in zip(tags, sentence_ids_and_tokens):
        sid, tokens = sentence_data
        output_entities(sid, tokens, sentence_tags)
