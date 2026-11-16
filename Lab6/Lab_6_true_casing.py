
"""
COSC 645 - Lab 6
True Casing Using Dynamic Programming

This program uses a simplified Viterbi algorithm
to restore capitalization in English sentences.
"""

import re
from pathlib import Path

# Words with preferred capitalization.
# These provide examples for the scoring function.
KNOWN_WORDS = {
    "i": "I",
    "john": "John",
    "smith": "Smith",
    "mary": "Mary",
    "jane": "Jane",
    "new": "New",
    "york": "York",
    "london": "London",
    "paris": "Paris",
    "microsoft": "Microsoft",
    "google": "Google",
    "python": "Python",
    "nasa": "NASA",
    "united": "United",
    "states": "States",
    "washington": "Washington",
    "baltimore": "Baltimore",
    "maryland": "Maryland",
}

# Test sentences
TEST_SENTENCES = [
    "i visited new york with john smith.",
    "mary and jane traveled to london.",
    "microsoft and google use artificial intelligence.",
    "i am learning python in my data mining class.",
    "nasa is located in the united states.",
    "john traveled from baltimore to washington.",
    "the students visited paris and new york.",
    "i live in maryland. john lives in new york.",
]


def generate_candidates(word):
    """Generate possible capitalization forms."""
    return list(dict.fromkeys([
        word.lower(),
        word.capitalize(),
        word.upper()
    ]))


def emission_score(original, candidate, is_start):
    """Score how appropriate a capitalization option is."""
    word = original.lower()

    if word in KNOWN_WORDS:
        score = 6.0 if candidate == KNOWN_WORDS[word] else -3.0
    elif candidate == word:
        score = 2.0
    elif candidate == word.capitalize():
        score = 0.0
    else:
        score = -2.0

    # Sentence beginnings normally use uppercase initials.
    if is_start:
        if candidate == word.capitalize():
            score += 4.0
        elif candidate == word:
            score -= 2.0

    return score


def transition_score(previous, current):
    """Score the transition between capitalization choices."""
    if previous.istitle() and current.istitle():
        return 0.3

    return 0.0


def truecase_segment(words):
    """Use Viterbi dynamic programming on a word sequence."""
    if not words:
        return []

    candidates = [generate_candidates(w) for w in words]

    # Dynamic programming scores and backpointers
    dp = []
    backpointers = []

    first_scores = {
        option: emission_score(words[0], option, True)
        for option in candidates[0]
    }

    dp.append(first_scores)
    backpointers.append({})

    for i in range(1, len(words)):
        current_scores = {}
        current_backpointers = {}

        for current in candidates[i]:
            best_score = float("-inf")
            best_previous = None

            for previous in candidates[i - 1]:
                score = (
                    dp[i - 1][previous]
                    + transition_score(previous, current)
                    + emission_score(words[i], current, False)
                )

                if score > best_score:
                    best_score = score
                    best_previous = previous

            current_scores[current] = best_score
            current_backpointers[current] = best_previous

        dp.append(current_scores)
        backpointers.append(current_backpointers)

    # Find the highest-scoring final option
    best_final = max(dp[-1], key=dp[-1].get)

    # Backtrack to reconstruct the best sequence
    result = [best_final]

    for i in range(len(words) - 1, 0, -1):
        best_final = backpointers[i][best_final]
        result.append(best_final)

    result.reverse()
    return result


def truecase_sentence(sentence):
    """Preserve punctuation while restoring capitalization."""
    tokens = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+",
        sentence
    )

    result = []
    pending_words = []
    pending_separators = []
    previous_separator = ""

    def flush_words():
        if pending_words:
            corrected = truecase_segment(pending_words)
            for i, word in enumerate(corrected):
                result.append(word)
                if i < len(pending_separators):
                    result.append(pending_separators[i])

            pending_words.clear()
            pending_separators.clear()

    for token in tokens:
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token):
            if re.search(r"[.!?]", previous_separator):
                flush_words()

            pending_words.append(token)
            previous_separator = ""

        else:
            if pending_words:
                pending_separators.append(token)
            else:
                result.append(token)

            previous_separator = token

    flush_words()
    return "".join(result)


def main():
    print("COSC 645 - Lab 6: True Casing")
    print("-" * 50)

    output_lines = [
        "COSC 645 - Lab 6: True Casing",
        "Dynamic Programming / Viterbi Algorithm",
        "-" * 50,
    ]

    for index, sentence in enumerate(TEST_SENTENCES, start=1):
        corrected = truecase_sentence(sentence)

        block = [
            f"\nTest {index}",
            f"Original:  {sentence}",
            f"Corrected: {corrected}",
        ]

        for line in block:
            print(line)

        output_lines.extend(block)

    output_path = (
        Path(__file__).resolve().parent
        / "Schuster_True_Casing_results.txt"
    )

    output_path.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8"
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
