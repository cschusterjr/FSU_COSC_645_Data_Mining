# COSC 645 - Lab 4: Person Name Extraction Using Name Corpora

## Overview

This lab builds on Lab 3 by using name corpora to improve the identification of person names in text. The goal is to develop a simple Python program that extracts possible person names from the first 10,000 sentences of the BNC Baby corpus.

Unlike Lab 3, which primarily used regular expressions and capitalization patterns, Lab 4 uses two reference files containing given names and surnames to help identify potential matches.

## Dataset

The program uses three files:

- `sents_BNCbaby.txt`: Sentences from the BNC Baby corpus.
- `given_names.txt`: A reference list of possible given names.
- `surnames.txt`: A reference list of possible surnames.

The program reads the first 10,000 sentences and compares words against the name corpora.

## Methodology

The program uses a heuristic approach based on three rules:

1. **Title + Given Name + Surname:** Identifies names such as "Sir James Goldsmith."
2. **Title + Surname:** Identifies names such as "Mr Franklin."
3. **Given Name + Surname:** Identifies names such as "Roland Franklin" and "Frank Kane."

Regular expressions are used to separate sentences into words. The program then checks capitalization and compares the words against the provided name lists.

Additional filtering removes some common false positives, while duplicate matches within individual sentences are excluded.

## Results

The program successfully processed 10,000 sentences and identified 3,776 possible person-name matches.

Examples include:

- Roland Franklin
- Frank Kane
- Sir James Goldsmith
- Mr Franklin
- Anthony Preece
- Robert Byrd

The extracted results are saved to `Schuster_extracted_person_names.txt`. Each match includes the sentence number where it was identified.

## Limitations

This approach has several limitations. Some words appear in the name corpora but may also refer to locations, organizations, or other terms. This can result in false positives.

The program may also miss names that are absent from the reference files or do not match the three extraction rules. For example, standalone first names are not independently extracted.

The program uses heuristic rules rather than a trained machine learning classifier. Although the results demonstrate how name corpora can improve text extraction, the matches have not been manually validated for accuracy.

## Running the Program

1. Install Python 3.
2. Place the Python script and the three input files in the same folder.
3. Open a terminal in the Lab4 directory.
4. Run:

   `python Lab_4_person_name_extractor.py`

5. Review the terminal output and the generated `Schuster_extracted_person_names.txt` file.

## Files

- `Lab_4_person_name_extractor.py`: Python source code.
- `Schuster_extracted_person_names.txt`: Extracted person-name matches.
- `given_names.txt`: Given-name reference corpus.
- `surnames.txt`: Surname reference corpus.
- `sents_BNCbaby.txt`: BNC Baby sentence dataset.
- `README.md`: Project documentation.

## Conclusion

Lab 4 demonstrates how reference name corpora can be combined with regular expressions and simple heuristic rules to identify possible person names in text. Compared with Lab 3, this approach adds additional validation by checking words against known given names and surnames.

Although the program still produces some false positives and misses certain names, it provides a straightforward example of using reference data to support information extraction.
