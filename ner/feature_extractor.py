"""Feature extraction for CRF-based Named Entity Recognition.

Extracts lexical, morphological, and contextual features from tokenized
biomedical text for training a CRF sequence labeler.
"""

import argparse
from os import listdir
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from xml.dom.minidom import parse

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
SIMPLE_DB_PATH = RESOURCES_DIR / "HSDB.txt"
DRUG_BANK_PATH = RESOURCES_DIR / "DrugBank.txt"

SimpleDrugDb: Set[str] = set()
DrugBank: Dict[str, Set[str]] = {"drug": set(), "brand": set(), "group": set()}


def read_drug_list_files() -> None:
    """Load drug name databases from resource files into module-level sets."""
    global SimpleDrugDb, DrugBank
    with open(SIMPLE_DB_PATH, "r") as f:
        SimpleDrugDb = {line.strip().lower() for line in f}

    with open(DRUG_BANK_PATH, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 2:
                name, n_type = parts[0].lower(), parts[1]
                if n_type in DrugBank:
                    DrugBank[n_type].add(name)


def tokenize(s: str) -> List[Tuple[str, int, int]]:
    """Tokenize a sentence and compute character offsets.

    Returns:
        List of (word, offset_from, offset_to) tuples.
    """
    token_list = []
    tokens = word_tokenize(s)
    search_start = 0
    for t in tokens:
        offset_from = s.find(t, search_start)
        offset_to = offset_from + len(t) - 1
        token_list.append((t, offset_from, offset_to))
        search_start = offset_from + 1
    return token_list


def get_tag(
    token: Tuple[str, int, int], gold: List[Tuple[int, int, str]]
) -> str:
    """Determine the BIO tag for a token given gold-standard entity spans.

    Returns:
        BIO tag string (e.g., 'B-drug', 'I-brand', 'O').
    """
    _, start, end = token
    for offset_from, offset_to, entity_type in gold:
        if start == offset_from and end <= offset_to:
            return "B-" + entity_type
        elif start > offset_from and end <= offset_to:
            return "I-" + entity_type
    return "O"


def has_numbers(word: str) -> bool:
    return any(c.isdigit() for c in word)


def num_digits(word: str) -> int:
    return sum(c.isdigit() for c in word)


def use_db_resources(word: str) -> Tuple[bool, str]:
    """Check if a word matches any entry in the drug databases."""
    lower = word.lower()
    if lower in SimpleDrugDb:
        return True, "drug"
    elif lower in DrugBank["drug"]:
        return True, "drug"
    elif lower in DrugBank["brand"]:
        return True, "brand"
    elif lower in DrugBank["group"]:
        return True, "group"
    return False, ""


def extract_features(
    tokenized_sentence: List[Tuple[str, int, int]], should_look_up: bool = False
) -> List[List[str]]:
    """Extract feature vectors for each token in a sentence.

    Features include surface form, suffixes/prefixes, POS tags, lemmas,
    and contextual features from neighboring tokens.

    Returns:
        List of sparse feature vectors (one per token).
    """
    punct = [".", ",", ";", ":", "?", "!"]

    # Batch POS-tag the full sentence for better accuracy
    words = [t[0] for t in tokenized_sentence]
    pos_tags = pos_tag(words) if words else []

    features = []
    for i, (t, _, _) in enumerate(tokenized_sentence):
        token_tag = pos_tags[i][1] if i < len(pos_tags) else "UNK"

        token_features = [
            "form=" + t,
            "formlower=" + t.lower(),
            "suf3=" + t[-3:],
            "suf4=" + t[-4:],
            "suf5=" + t[-5:],
            "prfx3=" + t[:3],
            "prfx4=" + t[:4],
            "prfx5=" + t[:5],
            "capitalized=%s" % t.istitle(),
            "uppercase=%s" % t.isupper(),
            "digit=%s" % t.isdigit(),
            "stopword=%s" % (t in STOP_WORDS),
            "punctuation=%s" % (t in punct),
            "length=%s" % len(t),
            "posTag=%s" % token_tag,
            "lemma=%s" % LEMMATIZER.lemmatize(t),
            "numDigits=%s" % num_digits(t),
            "containsDash=%s" % ("-" in t),
        ]

        if should_look_up:
            is_drug, is_type = use_db_resources(t)
            token_features.append("Ruled=%s" % (is_type if is_drug else "O"))

        features.append(token_features)

    # Add context features (previous/next token)
    for i, current_token in enumerate(features):
        if i > 0:
            prev_token = features[i - 1][0][5:]  # strip "form=" prefix
            current_token.append("prev=%s" % prev_token)
            current_token.append("suf3Prev=%s" % prev_token[-3:])
            current_token.append("suf4Prev=%s" % prev_token[-4:])
            current_token.append("prevIsTitle=%s" % prev_token.istitle())
            current_token.append("prevIsUpper=%s" % prev_token.isupper())
            current_token.append("PrevIsDigit=%s" % prev_token.isdigit())
        else:
            current_token.append("prev=_BoS_")

        if i < len(features) - 1:
            next_token = features[i + 1][0][5:]  # strip "form=" prefix
            current_token.append("next=%s" % next_token)
            current_token.append("suf3Next=%s" % next_token[-3:])
            current_token.append("suf4Next=%s" % next_token[-4:])
            current_token.append("NextIsTitle=%s" % next_token.istitle())
            current_token.append("NextIsUpper=%s" % next_token.isupper())
            current_token.append("NextIsDigit=%s" % next_token.isdigit())
        else:
            current_token.append("next=_EoS_")

    return features


def feature_extractor(
    datadir: str, resultpath: str, should_look_up: bool = False
) -> None:
    """Extract features from all XML files and write to output file.

    Args:
        datadir: Path to directory with XML-annotated files.
        resultpath: Path to write the feature file.
        should_look_up: Whether to include drug database lookup features.
    """
    if should_look_up:
        read_drug_list_files()

    with open(resultpath, "w") as result_f:
        for f in listdir(datadir):
            tree = parse(datadir + "/" + f)
            sentences = tree.getElementsByTagName("sentence")

            for s in sentences:
                sid = s.attributes["id"].value
                stext = s.attributes["text"].value

                # Load ground truth entities
                gold = []
                entities = s.getElementsByTagName("entity")
                for e in entities:
                    offset = e.attributes["charOffset"].value
                    # Handle discontinuous entities: take only the first span
                    first_span = offset.split(";")[0]
                    parts = first_span.split("-")
                    if len(parts) == 2:
                        start, end = int(parts[0]), int(parts[1])
                        gold.append((start, end, e.attributes["type"].value))

                tokens = tokenize(stext)
                features = extract_features(tokens, should_look_up)

                for i in range(len(tokens)):
                    tag = get_tag(tokens[i], gold)
                    joined_features = "\t".join(features[i])
                    result_f.write(
                        f"{sid}\t{tokens[i][0]}\t{tokens[i][1]}\t{tokens[i][2]}"
                        f"\t{tag}\t{joined_features}\n"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features for CRF-based NER."
    )
    parser.add_argument("data_to_extract_path", type=str, help="Path to data directory")
    parser.add_argument("output_file_name", type=str, help="Output file path")
    parser.add_argument(
        "-l",
        nargs="?",
        const=True,
        default=False,
        help="Include drug database lookup features",
    )
    args = parser.parse_args()
    feature_extractor(args.data_to_extract_path, args.output_file_name, args.l)
