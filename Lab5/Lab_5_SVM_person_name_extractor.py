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


# ---------------------------------------------------------
# Extract Word Features for SVM
# ---------------------------------------------------------

def extract_word_features(words, index):
    word = words[index]

    features = {
        "word.lower": word.lower(),
        "word.istitle": word.istitle(),
        "word.isupper": word.isupper(),
        "word.isdigit": word.isdigit(),
        "prefix2": word[:2].lower(),
        "suffix2": word[-2:].lower(),
        "suffix3": word[-3:].lower(),
        "length": len(word)
    }

    if index > 0:
        previous = words[index - 1]
        features["previous.lower"] = previous.lower()
        features["previous.istitle"] = previous.istitle()
    else:
        features["beginning"] = True

    if index < len(words) - 1:
        following = words[index + 1]
        features["next.lower"] = following.lower()
        features["next.istitle"] = following.istitle()
    else:
        features["ending"] = True

    return features


# ---------------------------------------------------------
# Prepare Training and Testing Data
# ---------------------------------------------------------

train_sentences, test_sentences = train_test_split(
    conll_sentences,
    test_size=0.20,
    random_state=645
)


def prepare_features(sentences):
    X = []
    y = []

    for sentence in sentences:
        words = [word for word, tag in sentence]

        for index, (word, tag) in enumerate(sentence):
            X.append(extract_word_features(words, index))
            y.append(tag)

    return X, y


X_train, y_train = prepare_features(train_sentences)
X_test, y_test = prepare_features(test_sentences)

vectorizer = DictVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)