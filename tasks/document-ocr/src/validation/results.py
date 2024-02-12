from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import csv


def avg(values: list[float]) -> float:
    if len(values) == 0:
        return 0

    return sum(values) / len(values)


@dataclass
class LabelAccuracyResult:

    name: str
    accuracy: float
    iou: float

    def displayValue(self) -> str:
        return f"{self.accuracy:.2f} ({self.iou:.2f})"


@dataclass
class SampleAccuracyResult:

    id: int
    name: str
    labelResults: list[LabelAccuracyResult]

    def getLabel(self, label: str) -> Optional[LabelAccuracyResult]:
        for labelResult in self.labelResults:
            if labelResult.name == label:
                return labelResult

        return None

    def getLabelDisplayValue(self, label: str) -> Optional[str]:
        for labelResult in self.labelResults:
            if labelResult.name == label:
                return labelResult.displayValue()

        return None

    def getAccuracy(self) -> float:
        return avg([result.accuracy for result in self.labelResults])

    def getIoU(self) -> float:
        return avg([result.iou for result in self.labelResults])

    def displayValue(self) -> str:
        return f"{self.getAccuracy():.2f} ({self.getIoU():.2f})"


@dataclass
class DatasetAccuracyResult:

    id: int
    name: str
    sampleResults: list[SampleAccuracyResult]

    def getLabelDisplayValue(self, name: str) -> Optional[str]:
        accuracy: list[float] = []
        iou: list[float] = []

        for sampleResult in self.sampleResults:
            label = sampleResult.getLabel(name)
            if label is None:
                continue

            accuracy.append(label.accuracy)
            iou.append(label.iou)

        return f"{avg(accuracy):.2f} ({avg(iou):.2f})"

    def displayValue(self) -> str:
        accuracy = sum(result.getAccuracy() for result in self.sampleResults) / len(self.sampleResults)
        iou = sum(result.getIoU() for result in self.sampleResults) / len(self.sampleResults)

        return f"{accuracy:.2f} ({iou:.2f})"

    def writeSampleResults(self, path: Path) -> None:
        with path.open("w") as file:
            writer = csv.DictWriter(file, ["id", "name", "first_name", "last_name", "date_of_birth", "gender", "total"])
            writer.writeheader()

            for sampleResult in self.sampleResults:
                firstName = sampleResult.getLabelDisplayValue("first_name")
                lastName = sampleResult.getLabelDisplayValue("last_name")
                birthDate = sampleResult.getLabelDisplayValue("date_of_birth")
                gender = sampleResult.getLabelDisplayValue("gender")

                row = {
                    "id": sampleResult.id,
                    "name": sampleResult.name,
                    "first_name": firstName if firstName is not None else "-",
                    "last_name": lastName if lastName is not None else "-",
                    "date_of_birth": birthDate if birthDate is not None else "-",
                    "gender": gender if gender is not None else "-",
                    "total": sampleResult.displayValue()
                }

                writer.writerow(row)

    def writeDatasetResult(self, path: Path) -> None:
        with path.open("w") as file:
            writer = csv.DictWriter(file, ["id", "name", "first_name", "last_name", "date_of_birth", "gender", "total"])
            writer.writeheader()

            firstName = self.getLabelDisplayValue("first_name")
            lastName = self.getLabelDisplayValue("last_name")
            birthDate = self.getLabelDisplayValue("date_of_birth")
            gender = self.getLabelDisplayValue("gender")

            row = {
                "id": self.id,
                "name": self.name,
                "first_name": firstName if firstName is not None else "-",
                "last_name": lastName if lastName is not None else "-",
                "date_of_birth": birthDate if birthDate is not None else "-",
                "gender": gender if gender is not None else "-",
                "total": self.displayValue()
            }

            writer.writerow(row)
