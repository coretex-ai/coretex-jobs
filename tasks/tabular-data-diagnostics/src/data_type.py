from enum import Enum


class DataType(Enum):

    csv = 0
    tsv = 1

    @property
    def extension(self) -> str:
        return self.name

    @property
    def delimiter(self) -> str:
        if self == DataType.csv:
            return ","

        if self == DataType.tsv:
            return "\t"

        raise ValueError("Invalid data type!")
