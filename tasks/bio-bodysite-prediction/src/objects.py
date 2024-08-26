from typing import Optional, NamedTuple

from scipy import sparse

import numpy as np


class Taxon:

    def __init__(self, taxonId: str, count: int):
        self.taxonId = taxonId
        self.count = count


class Sample:

    def __init__(self, sampleId: str, bodySite: str, associationSite: str, taxons: list[Taxon] = []) -> None:
        self.sampleId = sampleId
        self.bodySite = bodySite
        self.associationSite = associationSite
        self.taxons = taxons

    def addTaxon(self, taxon: Taxon) -> None:
        self.taxons.append(taxon)


class MatrixTuple(NamedTuple):

    inputMatrix: sparse.csr_matrix
    outputMatrix: np.ndarray
    sampleIdList: list[str]
    uniqueBodySite: dict[str, int]
    uniqueTaxons: dict[str, int]


class JsonTuple(NamedTuple):

    sampleData: list[Sample]
    uniqueTaxons: dict[str, int]
    uniqueBodySite: dict[str, int]
