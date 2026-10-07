# COSC 645 - Lab 4
# Person Name Extractor Using Name Corpora
# Charles Schuster

import re


# --------------------
# Program Settings
# --------------------

sentence_file = "sents_BNCbaby.txt"
given_names_file = "given_names.txt"
surnames_file = "surnames.txt"
output_file = "extracted_person_names.txt"
max_sentences = 10000


# --------------------
# Load Given Names
# --------------------

given_names = set()

with open(given_names_file, "r", encoding="utf-8") as file:
    for line in file:
        name = line.strip().lower()

        if name:
            given_names.add(name)


# --------------------
# Load Surnames
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

    # Look for a given name followed by a surname.
    # Examples: Roland Franklin, Frank Kane
    for i in range(len(words) - 1):

        first_word = words[i]
        second_word = words[i + 1]

        # Both words should be capitalized.
        if first_word[0].isupper() and second_word[0].isupper():

            # Check the words against the name corpora.
            if (
                first_word.lower() in given_names
                and second_word.lower() in surnames
            ):
                names.append(f"{first_word} {second_word}")

    # Look for a surname following a personal title.
    # Examples: Mr Franklin, Dr Smith, Professor Jones
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

    for i in range(len(words) - 1):

        title = words[i]
        next_word = words[i + 1]

        if (
            title in titles
            and next_word[0].isupper()
            and next_word.lower() in surnames
        ):
            names.append(f"{title} {next_word}")

    # Remove duplicate names while preserving their order.
    unique_names = []

    for name in names:
        if name not in unique_names:
            unique_names.append(name)

    return unique_names


# --------------------
# Test the Extractor
# --------------------

print("\nTesting name extraction on the first 30 sentences:")

for sentence_number, sentence in sentences[:30]:

    names = extract_person_names(sentence)

    if names:
        print(f"Sentence {sentence_number}: {names}")