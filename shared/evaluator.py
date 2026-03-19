"""Evaluation module for NER and DDI tasks.

Computes precision, recall, and F1 scores by comparing predicted entities
or interactions against gold-standard annotations in XML format.
"""

import argparse
import sys
from os import listdir
from typing import Dict, Set, Tuple
from xml.dom.minidom import parse


def add_instance(instance_set: Dict[str, Set[str]], einfo: str, etype: str) -> None:
    """Register an entity/relation instance in the evaluation set."""
    instance_set["CLASS"].add(einfo + "|" + etype)
    instance_set["NOCLASS"].add(einfo)
    if etype not in instance_set:
        instance_set[etype] = set()
    instance_set[etype].add(einfo)


def load_gold_NER(golddir: str) -> Dict[str, Set[str]]:
    """Load gold-standard NER entities from XML files in a directory."""
    entities: Dict[str, Set[str]] = {"CLASS": set(), "NOCLASS": set()}
    for f in listdir(golddir):
        tree = parse(golddir + "/" + f)
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value
            ents = s.getElementsByTagName("entity")
            for e in ents:
                einfo = (
                    sid
                    + "|"
                    + e.attributes["charOffset"].value
                    + "|"
                    + e.attributes["text"].value
                )
                etype = e.attributes["type"].value
                add_instance(entities, einfo, etype)
    return entities


def load_gold_DDI(golddir: str) -> Dict[str, Set[str]]:
    """Load gold-standard DDI relations from XML files in a directory."""
    relations: Dict[str, Set[str]] = {"CLASS": set(), "NOCLASS": set()}
    for f in listdir(golddir):
        tree = parse(golddir + "/" + f)
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                ddi = p.attributes["ddi"].value
                if ddi == "true":
                    rtype = p.attributes["type"].value
                    rinfo = sid + "|" + id_e1 + "|" + id_e2
                    add_instance(relations, rinfo, rtype)
    return relations


def load_predicted(task: str, outfile: str) -> Dict[str, Set[str]]:
    """Load predicted entities/relations from an output file."""
    predicted: Dict[str, Set[str]] = {"CLASS": set(), "NOCLASS": set()}
    with open(outfile, "r") as outf:
        for line in outf:
            line = line.strip()
            if not line:
                continue
            if line in predicted["CLASS"]:
                print("Ignoring duplicated entity in system predictions file: " + line)
                continue
            etype = line.split("|")[-1]
            einfo = "|".join(line.split("|")[:-1])
            add_instance(predicted, einfo, etype)
    return predicted


def statistics(
    gold: Dict[str, Set[str]], predicted: Dict[str, Set[str]], kind: str
) -> Tuple[int, int, int, int, int, float, float, float]:
    """Compute TP, FP, FN, precision, recall, and F1 for a given entity type."""
    tp = 0
    fp = 0
    nexp = len(gold[kind])

    if kind in predicted:
        npred = len(predicted[kind])
        for p in predicted[kind]:
            if p in gold[kind]:
                tp += 1
            else:
                fp += 1
        fn = sum(1 for p in gold[kind] if p not in predicted[kind])
    else:
        npred = 0
        fn = nexp

    P = tp / npred if npred != 0 else 0
    R = tp / nexp if nexp != 0 else 0
    F1 = 2 * P * R / (P + R) if P + R != 0 else 0
    return tp, fp, fn, npred, nexp, P, R, F1


def _row(txt: str) -> str:
    """Right-pad text to 17 characters for table alignment."""
    return txt + " " * (17 - len(txt))


def print_statistics(gold: Dict[str, Set[str]], predicted: Dict[str, Set[str]]) -> None:
    """Print a formatted evaluation table with per-class and aggregate metrics."""
    print(_row("") + "  tp\t  fp\t  fn\t#pred\t#exp\tP\tR\tF1")
    print("-" * 78)

    nk, sP, sR, sF1 = 0, 0.0, 0.0, 0.0
    for kind in sorted(gold):
        if kind in ("CLASS", "NOCLASS"):
            continue
        tp, fp, fn, npred, nexp, P, R, F1 = statistics(gold, predicted, kind)
        print(
            _row(kind)
            + "{:>4}\t{:>4}\t{:>4}\t{:>4}\t{:>4}\t{:2.1%}\t{:2.1%}\t{:2.1%}".format(
                tp, fp, fn, npred, nexp, P, R, F1
            )
        )
        nk, sP, sR, sF1 = nk + 1, sP + P, sR + R, sF1 + F1

    sP, sR, sF1 = sP / nk, sR / nk, sF1 / nk
    print("-" * 78)
    print(
        _row("M.avg")
        + "-\t-\t-\t-\t-\t{:2.1%}\t{:2.1%}\t{:2.1%}".format(sP, sR, sF1)
    )
    print("-" * 78)

    tp, fp, fn, npred, nexp, P, R, F1 = statistics(gold, predicted, "CLASS")
    print(
        _row("m.avg")
        + "{:>4}\t{:>4}\t{:>4}\t{:>4}\t{:>4}\t{:2.1%}\t{:2.1%}\t{:2.1%}".format(
            tp, fp, fn, npred, nexp, P, R, F1
        )
    )

    tp, fp, fn, npred, nexp, P, R, F1 = statistics(gold, predicted, "NOCLASS")
    print(
        _row("m.avg(no class)")
        + "{:>4}\t{:>4}\t{:>4}\t{:>4}\t{:>4}\t{:2.1%}\t{:2.1%}\t{:2.1%}".format(
            tp, fp, fn, npred, nexp, P, R, F1
        )
    )


def evaluate(task: str, golddir: str, outfile: str) -> None:
    """Run full evaluation pipeline: load gold + predictions, print metrics.

    Args:
        task: Either 'NER' or 'DDI'.
        golddir: Path to directory containing gold-standard XML files.
        outfile: Path to file containing system predictions.
    """
    if task == "NER":
        gold = load_gold_NER(golddir)
    elif task == "DDI":
        gold = load_gold_DDI(golddir)
    else:
        print(f"Invalid task '{task}'. Please specify 'NER' or 'DDI'.")
        sys.exit(1)

    predicted = load_predicted(task, outfile)
    print_statistics(gold, predicted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER or DDI predictions.")
    parser.add_argument("task", choices=["NER", "DDI"], help="Task type")
    parser.add_argument("golddir", help="Directory with gold-standard XML files")
    parser.add_argument("outfile", help="File with system predictions")
    args = parser.parse_args()
    evaluate(args.task, args.golddir, args.outfile)
