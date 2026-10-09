# COSC 645 - Lab 4
# Person Name Extractor Using Name Corpora
# Charles Schuster

import re
import csv
import random

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# --------------------
# Program Settings
# --------------------

sentence_file = "sents_BNCbaby.txt"
given_names_file = "given_names.txt"
surnames_file = "surnames.txt"
output_file = "Schuster_extracted_person_names.txt"
max_sentences = 10000


# --------------------
# Load Given Names File
# --------------------

given_names = set()

with open(given_names_file, "r", encoding="utf-8") as file:
    for line in file:
        name = line.strip().lower()

        if name:
            given_names.add(name)


# --------------------
# Load Surnames File
# --------------------

surnames = set()

with open(surnames_file, "r", encoding="utf-8") as file:
    for line in file:
        name = line.strip().lower()

        if name:
            surnames.add(name)


# --------------------
# Read the BNC Baby Dataset
# --------------------

sentences = []

with open(sentence_file, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file):

        if line_number >= max_sentences:
            break

        line = line.strip()

        # Dataset format:
        # SentenceNumber[TAB]EnglishSentence
        parts = line.split("\t", 1)

        if len(parts) == 2:
            sentence_number = parts[0]
            sentence = parts[1]

            sentences.append((sentence_number, sentence))


# --------------------
# Verify Files Loaded Correctly
# --------------------

print(f"Given names loaded: {len(given_names)}")
print(f"Surnames loaded: {len(surnames)}")
print(f"Sentences loaded: {len(sentences)}")

print("\nFirst 3 sentences:")

for sentence_number, sentence in sentences[:3]:
    print(f"Sentence {sentence_number}: {sentence}")


# --------------------
# Extract Person Names
# --------------------

def extract_person_names(sentence):
    """
    Identify possible person names using the given-name
    and surname corpora.
    """

    names = []

    # Split the sentence into words while keeping
    # alphabetic words and apostrophes.
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence)

    # Common personal titles.
    titles = {
        "Mr",
        "Mrs",
        "Ms",
        "Miss",
        "Dr",
        "Prof",
        "Professor",
        "Sir",
        "Lord",
        "Lady"
    }

    # Some words may appear in the given-name corpus even
    # though they are commonly used as titles or other words.
    excluded_first_words = {
        "The",
        "A",
        "An",
        "This",
        "That",
        "These",
        "Those",
        "Mr",
        "Mrs",
        "Ms",
        "Miss",
        "Dr",
        "Prof",
        "Professor",
        "Sir",
        "Lord",
        "Lady"
    }

    # --------------------
    # Rule 1:Title + Given Name + Surname
    # Example: Sir James Goldsmith
    # --------------------

    for i in range(len(words) - 2):

        title = words[i]
        first_name = words[i + 1]
        last_name = words[i + 2]

        if (
            title in titles
            and first_name[0].isupper()
            and last_name[0].isupper()
            and first_name.lower() in given_names
            and last_name.lower() in surnames
        ):
            names.append(
                f"{title} {first_name} {last_name}"
            )

    # --------------------
    # Rule 2: Title + Surname
    # Example: Mr Franklin
    # --------------------

    for i in range(len(words) - 1):

        title = words[i]
        next_word = words[i + 1]

        if (
            title in titles
            and next_word[0].isupper()
            and next_word.lower() in surnames
        ):
            # Do not add Title + Surname when it is actually
            # part of Title + Given Name + Surname.
            if (
                i + 2 < len(words)
                and next_word.lower() in given_names
                and words[i + 2][0].isupper()
                and words[i + 2].lower() in surnames
            ):
                continue

            names.append(f"{title} {next_word}")

    # --------------------
    # Rule 3: Given Name + Surname
    # Examples: Roland Franklin, Frank Kane
    # --------------------

    for i in range(len(words) - 1):

        first_word = words[i]
        second_word = words[i + 1]

        # Do not treat titles or common non-name words
        # as ordinary given names.
        if first_word in excluded_first_words:
            continue

        if (
            first_word[0].isupper()
            and second_word[0].isupper()
            and first_word.lower() in given_names
            and second_word.lower() in surnames
        ):
            # Avoid adding the two-word portion of a name
            # already captured with a title.
            if (
                i > 0
                and words[i - 1] in titles
            ):
                continue

            names.append(f"{first_word} {second_word}")



    # --------------------
    # Filter Common False Positives
    # --------------------

    excluded_phrases = {
        "square mile",
        "new york",
        "los angeles",
        "wall street",
        "the share",
        "west germany",
        "peat group"
    }

    names = [
        name for name in names
        if name.lower() not in excluded_phrases
    ]

    # --------------------
    # Remove Duplicates
    # --------------------

    unique_names = []

    for name in names:
        if name not in unique_names:
            unique_names.append(name)

    return unique_names


# --------------------
# Extract Features for Naive Bayes
# --------------------

def extract_name_features(name):
    words = name.split()
    lower_words = [word.lower() for word in words]

    first_word = lower_words[0]
    last_word = lower_words[-1]

    # Common titles associated with people
    titles = {
        "mr", "mrs", "ms", "miss", "dr",
        "prof", "professor", "sir", "lord", "lady"
    }

    # Words commonly associated with organizations or places
    organization_words = {
        "group", "company", "bank", "corporation",
        "university", "college", "office", "records",
        "council", "union", "league", "club",
        "gallery", "court", "policy", "minister"
    }

    location_words = {
        "street", "road", "avenue", "city", "town",
        "county", "state", "country", "island",
        "river", "park", "palace", "africa",
        "germany", "britain", "france", "london",
        "berlin", "thames"
    }

    common_words = {
        "the", "a", "an", "this", "that",
        "from", "first", "new", "old", "high",
        "south", "north", "east", "west",
        "prime", "national", "royal", "white"
    }

    features = [
        # Name dictionary features
        int(first_word in given_names),
        int(last_word in surnames),

        # Personal title features
        int(first_word in titles),
        int(any(word in titles for word in lower_words)),

        # Capitalization features
        int(all(word[0].isupper() for word in words)),
        int(name.isupper()),
        int(name.istitle()),

        # Organization and location features
        int(any(word in organization_words for word in lower_words)),
        int(any(word in location_words for word in lower_words)),

        # Common word features
        int(first_word in common_words),
        int(last_word in common_words),

        # Name structure features
        int(len(words) == 2),
        int(len(words) >= 3),

        # Additional dictionary checks
        int(
            first_word in given_names
            and last_word in surnames
        ),

        # Potential non-person phrases
        int(
            any(
                word in organization_words
                or word in location_words
                for word in lower_words
            )
        )
    ]

    return features


# --------------------
# Process the First 10,000 Sentences
# --------------------

results = []

for sentence_number, sentence in sentences:

    names = extract_person_names(sentence)

    for name in names:
        results.append((sentence_number, name))

# --------------------
# Train Bernoulli Naive Bayes
# --------------------

labeling_file = "Schuster_name_training_labels.csv"

training_names = []
training_labels = []

with open(labeling_file, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        training_names.append(row["name"])
        training_labels.append(int(row["is_person"]))

# Convert names into binary features.
X = [extract_name_features(name) for name in training_names]
y = training_labels

# Use 80% of examples for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=645,
    stratify=y
)

# Train the classifier.
model = BernoulliNB()
model.fit(X_train, y_train)

# Evaluate predictions on the test data.
predictions = model.predict(X_test)

print("\nNaive Bayes Classification Results:")
print(classification_report(y_test, predictions, zero_division=0))


# --------------------
# Apply Naive Bayes to Extracted Names
# --------------------

nb_results = []

for sentence_number, name in results:

    features = extract_name_features(name)

    prediction = model.predict([features])[0]

    probability = model.predict_proba([features])[0]

    # Probability associated with the person-name class.
    person_probability = probability[list(model.classes_).index(1)]

    nb_results.append(
        (sentence_number, name, prediction, person_probability)
    )


# --------------------
# Save Naive Bayes Results
# --------------------

nb_output_file = "Schuster_naive_bayes_results.txt"

with open(nb_output_file, "w", encoding="utf-8") as file:

    file.write("COSC 645 - Lab 4\n")
    file.write("Person Name Classification Using Bernoulli Naive Bayes\n")
    file.write("=" * 60 + "\n\n")

    for sentence_number, name, prediction, probability in nb_results:

        classification = "Person" if prediction == 1 else "Not Person"

        file.write(
            f"Sentence {sentence_number}: {name} | "
            f"{classification} | "
            f"Probability: {probability:.2f}\n"
        )

    file.write("\n" + "=" * 60 + "\n")

    total_persons = sum(
        1 for _, _, prediction, _ in nb_results
        if prediction == 1
    )

    file.write(f"Total candidates: {len(nb_results)}\n")
    file.write(f"Classified as persons: {total_persons}\n")
    file.write(
        f"Classified as non-persons: {len(nb_results) - total_persons}\n"
    )

print(f"\nNaive Bayes results saved to: {nb_output_file}")


# --------------------
# Display Sample Results
# --------------------

print("\nFirst 50 possible person names:")

for sentence_number, name in results[:50]:
    print(f"Sentence {sentence_number}: {name}")

print(f"\nTotal possible person names found: {len(results)}")

# --------------------
# Save Results to Output File
# --------------------

with open(output_file, "w", encoding="utf-8") as file:

    file.write("COSC 645 - Lab 4\n")
    file.write("Person Name Extraction Using Name Corpora\n")
    file.write("First 10,000 BNC Baby Sentences\n")
    file.write("=" * 50 + "\n\n")

    for sentence_number, name in results:
        file.write(f"Sentence {sentence_number}: {name}\n")

    file.write("\n" + "=" * 50 + "\n")
    file.write(
        f"Total possible person names found: {len(results)}\n"
    )


print(f"\nResults saved to: {output_file}")