"""Train a CRF model for Named Entity Recognition.

Uses python-crfsuite with L1/L2 regularization to learn sequence
labeling patterns from extracted feature files.
"""

import argparse

import pycrfsuite

from utils import read_feature_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CRF model for NER.")
    parser.add_argument("model_name", type=str, help="Output model file name")
    parser.add_argument("train_file_path", type=str, help="Path to training features")
    args = parser.parse_args()

    _, X_train, y_train = read_feature_file(args.train_file_path)

    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params(
        {
            "c1": 0.2,
            "c2": 0.001,
            "max_iterations": 1000,
            "feature.possible_states": True,
            "feature.possible_transitions": True,
        }
    )

    trainer.train(args.model_name)
