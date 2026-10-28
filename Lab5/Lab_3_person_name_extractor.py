# COSC 645 - Lab 3 Schuster 
# Person Name Extraction from the BNC Baby Corpus

import re

# ---------------------------------------------------------
# Program Settings
# ---------------------------------------------------------

input_file = "sents_BNCbaby.txt"
output_file = "extracted_person_names.txt"

# Lab 3 requires the first 10,000 sentences
max_sentences = 10000

# ---------------------------------------------------------
# Read the Dataset
# ---------------------------------------------------------

sentences = []

with open(input_file, "r", encoding="utf-8") as file:
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


print(f"Total sentences loaded: {len(sentences)}")


# ---------------------------------------------------------
# Person Name Extraction
# ---------------------------------------------------------

def extract_person_names(sentence):
    """
    Extract possible person names using simple
    capitalization-based heuristics and regular expressions.
    """

    names = []

    # Pattern 1: Find two or three consecutive capitalized words.
  
    full_name_pattern = (
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"
    )

    full_names = re.findall(full_name_pattern, sentence)
    names.extend(full_names)


    # Pattern 2: Find surnames following common personal titles.
    title_pattern = (
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Sir|Lord|Lady)"
        r"\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    )

    titled_names = re.findall(title_pattern, sentence)
    names.extend(titled_names)


    # Pattern 3: Find names written entirely in uppercase.
    uppercase_pattern = (
        r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,2}\b"
    )

    uppercase_names = re.findall(uppercase_pattern, sentence)

    for name in uppercase_names:
        names.append(name.title())

    # -----------------------------------------------------
    # Simple False-Positive Filter
    # -----------------------------------------------------

    # Capitalized phrases containing these words are more likely to represent organizations, locations, or other non-person entities.

    excluded_words = {
        "Investments",
        "Investment",
        "Exchange",
        "Airlines",
        "Airways",
        "Airport",
        "Association",
        "Corporation",
        "Company",
        "Companies",
        "Bank",
        "Insurance",
        "Securities",
        "University",
        "College",
        "Department",
        "Government",
        "Committee",
        "Council",
        "Commission",
        "Authority",
        "Group",
        "Limited",
        "Ltd",
        "International",
        "British",
        "American"
    }

    filtered_names = []

    for name in names:

        name_words = set(name.split())

        # Keep the name only if it does not contain one of the excluded words.
        if not name_words.intersection(excluded_words):
            filtered_names.append(name)

    # Remove duplicate names found within the same sentence while preserving their original order.

    unique_names = []

    for name in filtered_names:
        if name not in unique_names:
            unique_names.append(name)

    return unique_names

# ---------------------------------------------------------
# Process the First 10,000 Sentences
# ---------------------------------------------------------

results = []

for sentence_number, sentence in sentences:

    names = extract_person_names(sentence)

    for name in names:
        results.append((sentence_number, name))


# ---------------------------------------------------------
# Display Sample Results
# ---------------------------------------------------------

print("\nFirst 50 possible person names:")

for sentence_number, name in results[:50]:
    print(f"Sentence {sentence_number}: {name}")

print(f"\nTotal possible person names found: {len(results)}")


# ---------------------------------------------------------
# Save Results to Output File
# ---------------------------------------------------------

with open(output_file, "w", encoding="utf-8") as file:

    file.write("COSC 645 - Lab 3\n")
    file.write("Person Name Extraction from BNC Baby Corpus\n")
    file.write("First 10,000 Sentences\n")
    file.write("=" * 50 + "\n\n")

    for sentence_number, name in results:
        file.write(f"Sentence {sentence_number}: {name}\n")

    file.write("\n" + "=" * 50 + "\n")
    file.write(
        f"Total possible person names found: {len(results)}\n"
    )

print(f"\nResults saved to: {output_file}")