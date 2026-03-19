"""Rule-based Named Entity Recognition for drug names.

Classifies tokens as drug entities using suffix/prefix heuristics
and optional dictionary lookup against DrugBank and HSDB databases.
"""

import argparse
import sys
from os import listdir
from pathlib import Path
from typing import Dict, List, Set, Tuple
from xml.dom.minidom import parse
from xml.parsers.expat import ExpatError

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.evaluator import evaluate

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
SIMPLE_DB_PATH = RESOURCES_DIR / "HSDB.txt"
DRUG_BANK_PATH = RESOURCES_DIR / "DrugBank.txt"


class DrugDatabase:
    """Container for drug name databases used in dictionary lookup."""

    def __init__(self) -> None:
        self.simple_db: Set[str] = set()
        self.drug_bank: Dict[str, Set[str]] = {
            "drug": set(),
            "brand": set(),
            "group": set(),
        }

    def load(self) -> None:
        """Read drug databases from resource files."""
        with open(SIMPLE_DB_PATH, "r") as f:
            self.simple_db = {line.strip().lower() for line in f}

        with open(DRUG_BANK_PATH, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 2:
                    name, n_type = parts[0].lower(), parts[1]
                    if n_type in self.drug_bank:
                        self.drug_bank[n_type].add(name)


def tokenize(s: str) -> List[Tuple[str, int, int]]:
    """Tokenize a sentence and compute character offsets for each token.

    Filters out stopwords and non-alphabetic tokens to reduce workload.

    Returns:
        List of (word, offset_from, offset_to) tuples.
    """
    token_list = []
    tokens = word_tokenize(s)
    stop_words = set(stopwords.words("english"))
    search_start = 0
    for t in tokens:
        if t in stop_words or not t.isalpha():
            continue
        offset_from = s.find(t, search_start)
        offset_to = offset_from + len(t) - 1
        token_list.append((t, offset_from, offset_to))
        search_start = offset_from + 1
    return token_list


def token_type_classifier(
    word: str, drug_db: DrugDatabase = None, should_look_up: bool = False
) -> Tuple[bool, str]:
    """Classify a token as a drug entity type using heuristic rules.

    Args:
        word: Token text to classify.
        drug_db: Drug database for dictionary lookup.
        should_look_up: Whether to check against drug databases.

    Returns:
        Tuple of (is_entity, entity_type) where entity_type is one of
        'drug', 'brand', 'group', 'drug_n', or empty string.
    """
    threes = ["nol", "lol", "hol", "lam", "pam"]
    fours = ["arin", "oxin", "toin", "pine", "tine", "bital", "inol", "pram"]
    fives = ["azole", "idine", "orine", "mycin", "hrine", "exate", "amine", "emide"]

    drug_n = ["PCP", "18-MC", "methyl", "phenyl", "tokin", "fluo", "ethyl"]

    groups = [
        "depressants", "steroid", "ceptives", "urates", "amines", "azines",
        "phenones", "inhib", "coagul", "block", "acids", "agent", "+", "-",
        "NSAID", "TCA", "SSRI", "MAO",
    ]

    if should_look_up and drug_db:
        if word.lower() in drug_db.simple_db:
            return True, "drug"
        if word.lower() in drug_db.drug_bank["drug"]:
            return True, "drug"
        if word.lower() in drug_db.drug_bank["brand"]:
            return True, "brand"
        if word.lower() in drug_db.drug_bank["group"]:
            return True, "group"

    if word.isupper() and len(word) >= 4:
        return True, "brand"
    elif word[-3:] in threes or word[-4:] in fours or word[-5:] in fives:
        return True, "drug"
    elif any(t in word for t in groups) or (word[-1:] == "s" and len(word) >= 8):
        return True, "group"
    elif any(t in word for t in drug_n) or (word.isupper() and 2 <= len(word) < 4):
        return True, "drug_n"
    else:
        return False, ""


def extract_entities(
    sentence: List[Tuple[str, int, int]],
    drug_db: DrugDatabase = None,
    should_look_up: bool = False,
) -> List[Dict[str, str]]:
    """Identify drug entities from tokenized sentence using heuristic rules.

    Returns:
        List of entity dicts with keys 'name', 'offset', and 'type'.
    """
    output = []
    for token_text, offset_from, offset_to in sentence:
        is_entity, type_text = token_type_classifier(token_text, drug_db, should_look_up)
        if is_entity:
            entity = {
                "name": token_text,
                "offset": f"{offset_from}-{offset_to}",
                "type": type_text,
            }
            output.append(entity)
    return output


def main(datadir: str, outfile: str, should_look_up: bool = False) -> None:
    """Run rule-based NER on all XML files in a directory.

    Args:
        datadir: Path to directory with XML-annotated biomedical text files.
        outfile: Path to write entity predictions.
        should_look_up: Whether to use drug database lookup.
    """
    drug_db = DrugDatabase()
    if should_look_up:
        drug_db.load()

    with open(outfile, "w") as outf:
        for f in listdir(datadir):
            try:
                tree = parse(datadir + "/" + f)
            except ExpatError:
                continue

            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value
                stext = s.attributes["text"].value

                tokens = tokenize(stext)

                # create bigrams from consecutive tokens
                bigrams = filter(
                    lambda t: t[1][1] - t[0][2] == 2, zip(tokens[:-1], tokens[1:])
                )
                bigrams = [(f"{t0[0]} {t1[0]}", t0[1], t1[2]) for (t0, t1) in bigrams]
                tokens.extend(bigrams)

                entities = extract_entities(tokens, drug_db, should_look_up)

                for e in entities:
                    line = f'{sid}|{e["offset"]}|{e["name"]}|{e["type"]}'
                    outf.write(line + "\n")

    evaluate("NER", datadir, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based NER for drug names.")
    parser.add_argument("train_data_path", type=str, help="Path to data directory")
    parser.add_argument("output_file_name", type=str, help="Output file name")
    parser.add_argument(
        "-l",
        nargs="?",
        const=True,
        default=False,
        help="Enable dictionary lookup against drug databases",
    )
    args = parser.parse_args()
    main(args.train_data_path, args.output_file_name, args.l)
