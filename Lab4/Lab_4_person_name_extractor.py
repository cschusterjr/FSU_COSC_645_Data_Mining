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
    # Rule 3:
    # Given Name + Surname
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
        "the share"
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
# Process the First 10,000 Sentences
# --------------------

results = []

for sentence_number, sentence in sentences:

    names = extract_person_names(sentence)

    for name in names:
        results.append((sentence_number, name))

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