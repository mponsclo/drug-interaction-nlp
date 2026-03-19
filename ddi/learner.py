"""Train a Maximum Entropy classifier for Drug-Drug Interaction detection.

Uses NLTK's MaxentClassifier with the MEGAM optimization backend.

Note: Requires MEGAM binary. Set the MEGAM_PATH environment variable
to point to the megam executable, or pass it as a CLI argument.
"""

import argparse
import os

import nltk.classify

from utils import read_feature_file, read_test_feature_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaxEnt DDI classifier.")
    parser.add_argument("train_file_path", type=str, help="Path to training features")
    parser.add_argument("test_file_path", type=str, help="Path to test features")
    parser.add_argument("output_file_path", type=str, help="Path for predictions output")
    parser.add_argument(
        "--megam-path",
        type=str,
        default=os.environ.get("MEGAM_PATH", "megam"),
        help="Path to MEGAM binary (default: $MEGAM_PATH or 'megam')",
    )
    args = parser.parse_args()

    train_data = read_feature_file(args.train_file_path)
    test_data = read_test_feature_file(args.test_file_path)

    nltk.classify.megam.config_megam(args.megam_path)
    model = nltk.classify.MaxentClassifier.train(train_data, "megam")

    with open(args.output_file_path, "w") as output_f:
        for sid, e1, e2, feats in test_data:
            prediction = model.classify(feats)
            if prediction != "null":
                output_f.write(f"{sid}|{e1}|{e2}|{prediction}\n")
