# COSC 645 - Lab 5
# Person Name Extraction Using Support Vector Machines
# Charles Schuster

import re
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


# ---------------------------------------------------------
# Program Settings
# ---------------------------------------------------------

folder = Path(__file__).resolve().parent

conll_file = folder / "Conll_NER_corpus.txt"
sentence_file = folder / "sents_BNCbaby.txt"
output_file = folder / "Schuster_SVM_results.txt"

max_sentences = 10000


# ---------------------------------------------------------
# Load CoNLL NER Training Data
# ---------------------------------------------------------

conll_sentences = []
current_sentence = []

with open(conll_file, "r", encoding="utf-8-sig") as file:
    for line in file:
        line = line.strip()

        if not line or line.startswith("#"):
            if current_sentence:
                conll_sentences.append(current_sentence)
                current_sentence = []
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        word = parts[0]
        tag = parts[-1]

        # Simplify the classification to person-name tags.
        if tag.lower().endswith("-per"):
            label = tag.upper()
        else:
            label = "O"

        if label not in {"B-PER", "I-PER"}:
            label = "O"

        current_sentence.append((word, label))

if current_sentence:
    conll_sentences.append(current_sentence)

print(f"CoNLL sentences loaded: {len(conll_sentences)}")
