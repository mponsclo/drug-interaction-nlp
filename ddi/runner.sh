#!/bin/bash
# DDI pipeline: feature extraction, training, and evaluation.
#
# Usage:
#   DATA_DIR=/path/to/semeval/data bash runner.sh
#
# Requires:
#   - Stanford CoreNLP server running on localhost:9000
#   - MEGAM binary (set MEGAM_PATH env var)

set -e

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the path containing train/ and test/ subdirectories}"

echo "Extracting test features..."
python3 feature_extractor.py "$DATA_DIR/test" > feats_test.dat

echo "Extracting train features..."
python3 feature_extractor.py "$DATA_DIR/train" > feats.dat

echo "Training and classifying..."
python3 learner.py feats.dat feats_test.dat output.dat

echo "Evaluating..."
python3 ../shared/evaluator.py DDI "$DATA_DIR/test" output.dat
