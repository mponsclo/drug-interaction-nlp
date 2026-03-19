"""Feature extraction for MaxEnt-based Drug-Drug Interaction detection.

Extracts syntactic and semantic features from dependency-parsed sentences
for training a Maximum Entropy classifier to detect DDI types.

Note: Requires a running Stanford CoreNLP server on localhost:9000.
"""

import argparse
import string
from os import listdir
from typing import Dict, List, Optional, Tuple
from xml.dom.minidom import parse

import networkx
from nltk import pos_tag
from nltk.parse.corenlp import CoreNLPDependencyParser

_corenlp_parser = None

CLUE_VERBS = [
    "administer", "enhance", "interact", "coadminister", "increase", "decrease"
]
NEGATIVE_WORDS = [
    "No", "not", "neither", "without", "lack", "fail", "unable", "abrogate",
    "absence", "prevent", "unlikely", "unchanged", "rarely", "inhibitor",
]


def _get_parser() -> CoreNLPDependencyParser:
    """Lazily initialize the CoreNLP dependency parser."""
    global _corenlp_parser
    if _corenlp_parser is None:
        _corenlp_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    return _corenlp_parser


def do_indices_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    return start1 == start2 and end1 == end2


def find_entity_in_tree(eid: str, entities: Dict, tree) -> Optional[Dict]:
    """Find the dependency tree node corresponding to an entity."""
    start_e1 = int(entities[eid]["offsets"][0])
    end_e1 = int(entities[eid]["offsets"][1].split(";")[0])

    for n in tree.nodes.items():
        node = n[1]
        if node["word"] and (node["start"] == start_e1 or node["end"] == end_e1):
            return node
    return None


def find_other_entities(
    eid1: str, eid2: str, sid: str, entities: Dict, tree
) -> List[Tuple[Optional[Dict], str]]:
    """Find all entities in the sentence other than the target pair."""
    other = [
        (entity["eid"], entity["type"])
        for _, entity in entities.items()
        if entity["sid"] == sid and entity["eid"] not in [eid1, eid2]
    ]
    return [(find_entity_in_tree(eid, entities, tree), e_type) for eid, e_type in other]


def get_offsets(word: str, s: str) -> Tuple[int, int]:
    """Find the start and end character offset of a word in a sentence."""
    start = s.find(word)
    end = start + len(word) - 1
    return start, end


def preprocess(s: str) -> str:
    """Escape characters that cause CoreNLP errors."""
    return s.replace("%", "<percentage>")


def analyze(s: str):
    """Parse a sentence with CoreNLP and enrich tokens with character offsets.

    Args:
        s: Sentence text.

    Returns:
        An nltk DependencyGraph enriched with start/end offsets per token.
    """
    s = s.replace("%", "<percentage>")
    parser = _get_parser()
    tree, = parser.raw_parse(s)
    for n in tree.nodes.items():
        node = n[1]
        if node["word"]:
            start, end = get_offsets(node["word"], s)
            node["start"] = start
            node["end"] = end
    return tree


def find_clue_verbs(path: List[int], tree) -> List[str]:
    """Check if any clue verbs appear on the dependency path."""
    path_nodes = [tree.nodes[x]["lemma"] for x in path]
    return [f"lemmainbetween={pn}" for pn in path_nodes if pn in CLUE_VERBS]


def negative_words_path(path: List[int], tree) -> int:
    """Count negative words along the dependency path."""
    path_nodes = [tree.nodes[x]["word"] for x in path]
    return sum(
        1 for pn in path_nodes
        if pn in NEGATIVE_WORDS or (pn and pn[-3:] == "n't")
    )


def negative_words_sentence(tree) -> int:
    """Count negative words in the entire sentence."""
    return sum(
        1 for n in tree.nodes.items()
        if n[1]["word"] in NEGATIVE_WORDS
    )


def traverse_path(path: List[int], tree) -> Tuple[Optional[str], Optional[str]]:
    """Traverse the dependency path between two entities.

    Returns:
        Tuple of (lemma_path, tag_path) string representations.
    """
    if not path:
        return None, None

    path_nodes = [tree.nodes[x] for x in path]
    str_path = ""

    # Traverse from e1 upward
    current_node = path_nodes[0]
    while current_node["head"] in path:
        rel = current_node["rel"]
        current_node = tree.nodes[current_node["head"]]
        str_path += rel + "<"

    tag_path = str_path + current_node["tag"]
    str_path += current_node["lemma"]

    # Traverse from e2 upward
    current_node = path_nodes[-1]
    while current_node["head"] in path:
        rel = current_node["rel"]
        current_node = tree.nodes[current_node["head"]]
        str_path += ">" + rel
        tag_path += ">" + rel

    return str_path, tag_path


def find_words_outside_path(
    path: List[int], tree
) -> Tuple[List[str], List[str]]:
    """Find lemmas before and after the dependency path in the sentence."""
    if len(path) < 1:
        return [], []

    words_before = []
    words_after = []
    nodes_before = [node[1] for node in tree.nodes.items()][:path[0]]
    nodes_after = [node[1] for node in tree.nodes.items()][path[-1]:]

    for node in nodes_before:
        if (
            node["address"] not in path
            and node["lemma"]
            and node["lemma"] not in string.punctuation
            and not node["lemma"].isdigit()
        ):
            words_before.append(node["lemma"])

    for node in nodes_after:
        if (
            node["address"] not in path
            and node["lemma"]
            and node["lemma"] not in string.punctuation
            and not node["lemma"].isdigit()
        ):
            words_after.append(node["lemma"])

    return words_before, words_after


def find_head(tree, entity: Dict) -> Optional[Dict]:
    """Find the head node of an entity in the dependency tree."""
    for n in tree.nodes.items():
        node = n[1]
        if node["address"] == entity["head"]:
            return node
    return None


def extract_features(
    tree, entities: Dict, e1: str, e2: str, sid: str
) -> List[str]:
    """Compute a sparse feature vector for a pair of entities.

    Features include head lemmas, POS tags, dependency path, negation,
    entity types, and contextual words outside the path.

    Args:
        tree: DependencyGraph with sentence analysis.
        entities: All entities in the sentence.
        e1, e2: Entity IDs for the target pair.
        sid: Sentence identifier.

    Returns:
        List of active features in sparse representation.
    """
    e1_node = find_entity_in_tree(e1, entities, tree)
    e2_node = find_entity_in_tree(e2, entities, tree)

    e1_head = find_head(tree, e1_node) if e1_node else None
    e2_head = find_head(tree, e2_node) if e2_node else None

    h1_lemma = e1_head["lemma"] if e1_head else None
    h2_lemma = e2_head["lemma"] if e2_head else None

    tag_head_e1 = e1_head["tag"] if e1_head else None
    tag_head_e2 = e2_head["tag"] if e2_head else None

    nxgraph = tree.nx_graph().to_undirected()
    shortest_path = (
        networkx.shortest_path(nxgraph, e1_node["address"], e2_node["address"])
        if (e1_node and e2_node)
        else []
    )
    path_with_word, path_with_tag = traverse_path(shortest_path, tree)
    count_neg_s = negative_words_sentence(tree)

    features = [
        f"h1_lemma={h1_lemma}",
        f"h2_lemma={h2_lemma}",
        f"h1_tag={tag_head_e1}",
        f"h2_tag={tag_head_e2}",
        f"tagpath={path_with_tag}",
        f"neg_words_s={count_neg_s}",
        f"e1_type={entities[e1]['type']}",
        f"e2_type={entities[e2]['type']}",
    ] + find_clue_verbs(shortest_path, tree)

    if e1_head and e2_head:
        if h1_lemma == h2_lemma:
            features.append("under_same=True")
            if tag_head_e1[0].lower() == "v":
                features.append("under_same_verb=True")
            else:
                features.append("under_same_verb=False")
        else:
            features.append("under_same=False")
            features.append("under_same_verb=False")

        if h1_lemma == e2_node["lemma"]:
            features.append("1under2=True")
        else:
            features.append("1under2=False")

        if h2_lemma == e1_node["lemma"]:
            features.append("2under1=True")
        else:
            features.append("2under1=False")

    words_before, words_after = find_words_outside_path(shortest_path, tree)
    for word in words_before:
        features.append(f"lemmabefore={word}")
        features.append(f"tagbefore={pos_tag([word])[0][1]}")
    for word in words_after:
        features.append(f"lemmaafter={word}")
        features.append(f"tagafter={pos_tag([word])[0][1]}")

    other_entities = find_other_entities(e1, e2, sid, entities, tree)
    for _, e_type in other_entities:
        features.append(f"typeother={e_type}")

    return features


def main(datadir: str) -> None:
    """Extract features from all XML files and print to stdout."""
    for f in listdir(datadir):
        tree = parse(datadir + "/" + f)
        sentences = tree.getElementsByTagName("sentence")

        for s in sentences:
            sid = s.attributes["id"].value
            stext = s.attributes["text"].value

            if len(stext) == 0:
                continue

            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                eid = e.attributes["id"].value
                entities[eid] = {
                    "offsets": e.attributes["charOffset"].value.split("-"),
                    "type": e.attributes["type"].value,
                    "sid": sid,
                    "eid": eid,
                }

            if len(entities) > 1:
                analysis = analyze(stext)

            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                ddi = p.attributes["ddi"].value
                dditype = p.attributes["type"].value if ddi == "true" else "null"

                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value

                feats = extract_features(analysis, entities, id_e1, id_e2, sid)
                print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features for DDI detection."
    )
    parser.add_argument("data_to_extract_path", type=str, help="Path to data directory")
    args = parser.parse_args()
    main(args.data_to_extract_path)
