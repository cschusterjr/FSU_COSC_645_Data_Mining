# COSC 645 - Lab 4
# Person Name Extractor Using Name Corpora
# Charles Schuster

#--------------------
# Program Settings
#--------------------

sentence_file = "sents_BNCbaby.txt"
given_names_file = "given_names.txt"
surnames_file = "surnames.txt"
output_file = "extracted_person_names.txt"
max_sentences = 10000

#--------------------
# Load Given Names
#-------------------
 
given_names = set()
with open(given_names_file, "r", encoding="utf-8") as file:
    for line in file:
        name = line.strip().lower()

        if name:
            given_names.add(name)

#--------------------
# Load Surnames
#--------------------

surnames = set()
with open(surnames_file, "r", encoding="utf-8") as file:
    for line in file:
        name = line.strip().lower()
        if name:
            surnames.add(name)

#--------------------
# Read the BNC Baby Dataset
#--------------------

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

#--------------------
# Verify files loaded correctly
#--------------------

print(f"Given names loaded: {len(given_names)}")
print(f"Surnames loaded: {len(surnames)}")
print(f"Sentences loaded: {len(sentences)}")

print("\nFirst 3 sentences:")

for sentence_number, sentence in sentences[:3]:
    print(f"Sentence {sentence_number}: {sentence}")