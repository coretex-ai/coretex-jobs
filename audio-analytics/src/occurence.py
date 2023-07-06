from typing import Optional
from typing_extensions import Self

from coretex.codable import Codable, KeyDescriptor


class SingleOccurrence(Codable):

    startIndex: int
    endIndex: int
    startTime: Optional[float]
    endTime: Optional[float]

    @classmethod
    def create(
        cls,
        startIndex: int,
        endIndex: int,
        startTime: Optional[float] = None,
        endTime: Optional[float] = None
    ) -> Self:

        obj = cls()

        obj.startIndex = startIndex
        obj.endIndex = endIndex
        obj.startTime = startTime
        obj.endTime = endTime

        return obj


class EntityOccurrence(Codable):

    text: str
    occurrences: list[SingleOccurrence]

    @classmethod
    def _keyDescriptors(cls) -> dict[str, KeyDescriptor]:
        descriptors = super()._keyDescriptors()
        descriptors["occurrences"] = KeyDescriptor("occurrences", SingleOccurrence, list)

        return descriptors

    @classmethod
    def create(cls, text: str) -> Self:
        obj = cls()

        obj.text = text
        obj.occurrences = []

        return obj

    def addOccurrence(
        self,
        startIndex: int,
        endIndex: int,
        startTime: Optional[float] = None,
        endTime: Optional[float] = None
    ) -> None:

        self.occurrences.append(SingleOccurrence.create(startIndex, endIndex, startTime, endTime))


class NamedEntityRecognitionResult(Codable):

    datasetId: int
    entityOccurrences: list[EntityOccurrence]

    @classmethod
    def _keyDescriptors(cls) -> dict[str, KeyDescriptor]:
        descriptors = super()._keyDescriptors()
        descriptors["entityOccurrences"] = KeyDescriptor("entityOccurrences", EntityOccurrence, list)

        return descriptors

    @classmethod
    def create(cls, datasetId: int, result: list[EntityOccurrence]) -> Self:
        obj = cls()

        obj.datasetId = datasetId
        obj.entityOccurrences = result

        return obj
