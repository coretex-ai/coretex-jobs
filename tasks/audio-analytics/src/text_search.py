from coretex.nlp import Token  # type: ignore[attr-defined]

from .occurence import EntityOccurrence


def findOccurrences(targetWord: str, tokens: list[Token]) -> EntityOccurrence:
    occurrence = EntityOccurrence.create(targetWord)

    for token in tokens:
        if token.text.upper() == targetWord.upper():
            occurrence.addOccurrence(token.startIndex, token.endIndex, token.startTime, token.endTime)

    return occurrence


def searchTranscription(tokens: list[Token], targetWords: list[str]) -> list[EntityOccurrence]:
    wordOccurrences: list[EntityOccurrence] = []

    for targetWord in targetWords:
        wordOccurrence = findOccurrences(targetWord, tokens)

        # No need to add empty occurrences
        if len(wordOccurrence.occurrences) == 0:
            continue

        wordOccurrences.append(wordOccurrence)

    return wordOccurrences
