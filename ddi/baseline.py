"""Rule-based Drug-Drug Interaction detection (stub).

This module provides the scaffolding for a rule-based DDI detector using
dependency parsing via Stanford CoreNLP. The check_interaction function
is intentionally left as a stub -- the ML and neural approaches in this
project provide the complete DDI classification pipelines.

Note: Requires a running Stanford CoreNLP server on localhost:9000.
"""

import argparse
import sys
from os import listdir
from pathlib import Path
from typing import Dict, Optional, Tuple
from xml.dom.minidom import parse

from nltk.parse.corenlp import CoreNLPDependencyParser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.evaluator import evaluate

_corenlp_parser = None


def _get_parser() -> CoreNLPDependencyParser:
    """Lazily initialize the CoreNLP dependency parser."""
    global _corenlp_parser
    if _corenlp_parser is None:
        _corenlp_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    return _corenlp_parser


def get_offsets(word: str, s: str) -> Tuple[int, int]:
    """Find the start and end character offset of a word in a sentence."""
    start = s.find(word)
    end = start + len(word) - 1
    return start, end


def analyze(s: str):
    """Parse a sentence with CoreNLP and enrich tokens with character offsets.

    Args:
        s: Sentence text.

    Returns:
        An nltk DependencyGraph enriched with start/end offsets per token.
    """
    parser = _get_parser()
    tree, = parser.raw_parse(s)
    for n in tree.nodes.items():
        node = n[1]
        if node["word"]:
            start, end = get_offsets(node["word"], s)
            node["start"] = start
            node["end"] = end
    return tree


def check_interaction(
    analysis, entities: Dict, e1: str, e2: str
) -> Optional[str]:
    """Determine if two entities have a drug-drug interaction.

    Stub implementation -- returns None (no interaction detected).
    A complete implementation would analyze the dependency path between
    the two entities and apply pattern-matching rules.

    Returns:
        Interaction type ('effect', 'mechanism', 'advice', 'int') or None.
    """
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based DDI detection (stub).")
    parser.add_argument("datadir", type=str, help="Path to data directory")
    parser.add_argument("output_file_name", type=str, help="Output file name")
    args = parser.parse_args()

    with open(args.output_file_name, "w") as outf:
        for f in listdir(args.datadir):
            tree = parse(args.datadir + "/" + f)
            sentences = tree.getElementsByTagName("sentence")

            for s in sentences:
                sid = s.attributes["id"].value
                stext = s.attributes["text"].value

                entities = {}
                ents = s.getElementsByTagName("entity")
                for e in ents:
                    eid = e.attributes["id"].value
                    entities[eid] = e.attributes["charOffset"].value.split("-")

                analysis = analyze(stext)

                pairs = s.getElementsByTagName("pair")
                for p in pairs:
                    id_e1 = p.attributes["e1"].value
                    id_e2 = p.attributes["e2"].value
                    ddi_type = check_interaction(analysis, entities, id_e1, id_e2)
                    if ddi_type is not None:
                        outf.write(f"{sid}|{id_e1}|{id_e2}|{ddi_type}\n")


if __name__ == "__main__":
    main()
