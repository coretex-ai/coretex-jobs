from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import csv

from coretex.utils import mathematicalRound


@dataclass
class LabelAccuracyResult:

    name: str
    accuracy: float


@dataclass
class SampleAccuracyResult:

    id: int
    name: str
    labelResults: list[LabelAccuracyResult]

    def getLabelAccuracy(self, label: str) -> Optional[float]:
        for labelResult in self.labelResults:
            if labelResult.name == label:
                return labelResult.accuracy

        return None

    def getAccuracy(self) -> float:
        return sum(result.accuracy for result in self.labelResults) / len(self.labelResults)


@dataclass
class DatasetAccuracyResult:

    id: int
    name: str
    sampleResults: list[SampleAccuracyResult]

    def getLabelAccuracy(self, label: str) -> Optional[float]:
        data: list[float] = []

        for sampleResult in self.sampleResults:
            labelAccuracy = sampleResult.getLabelAccuracy(label)
            if labelAccuracy is None:
                continue

            data.append(labelAccuracy)

        # Dataset doesn't have annotation for that label
        if len(data) == 0:
            return None

        return sum(data) / len(data)

    def getAccuracy(self) -> float:
        return sum(result.getAccuracy() for result in self.sampleResults) / len(self.sampleResults)

    def writeSampleResults(self, path: Path) -> None:
        with path.open("w") as file:
            writer = csv.DictWriter(file, ["id", "name", "first_name", "last_name", "date_of_birth", "gender", "total"])
            writer.writeheader()

            for sampleResult in self.sampleResults:
                firstName = sampleResult.getLabelAccuracy("first_name")
                lastName = sampleResult.getLabelAccuracy("last_name")
                birthDate = sampleResult.getLabelAccuracy("date_of_birth")
                gender = sampleResult.getLabelAccuracy("gender")

                row = {
                    "id": sampleResult.id,
                    "name": sampleResult.name,
                    "first_name": mathematicalRound(firstName, 2) if firstName is not None else "-",
                    "last_name": mathematicalRound(lastName, 2) if lastName is not None else "-",
                    "date_of_birth": mathematicalRound(birthDate, 2) if birthDate is not None else "-",
                    "gender": mathematicalRound(gender, 2) if gender is not None else "-",
                    "total": mathematicalRound(sampleResult.getAccuracy(), 2)
                }

                writer.writerow(row)

    def writeDatasetResult(self, path: Path) -> None:
        with path.open("w") as file:
            writer = csv.DictWriter(file, ["id", "name", "first_name", "last_name", "date_of_birth", "gender", "total"])
            writer.writeheader()

            firstName = self.getLabelAccuracy("first_name")
            lastName = self.getLabelAccuracy("last_name")
            birthDate = self.getLabelAccuracy("date_of_birth")
            gender = self.getLabelAccuracy("gender")

            row = {
                "id": self.id,
                "name": self.name,
                "first_name": mathematicalRound(firstName, 2) if firstName is not None else "-",
                "last_name": mathematicalRound(lastName, 2) if lastName is not None else "-",
                "date_of_birth": mathematicalRound(birthDate, 2) if birthDate is not None else "-",
                "gender": mathematicalRound(gender, 2) if gender is not None else "-",
                "total": mathematicalRound(self.getAccuracy(), 2)
            }

            writer.writerow(row)
