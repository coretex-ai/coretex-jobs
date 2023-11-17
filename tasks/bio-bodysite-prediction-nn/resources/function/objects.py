class Taxon:

    def __init__(self, taxonId: str, count: int):
        self.taxonId = taxonId
        self.count = count


class Sample:

    def __init__(self, sampleId: str, taxons: list[Taxon] = []) -> None:
        self.sampleId = sampleId
        self.taxons = taxons

    def addTaxon(self, taxon: Taxon) -> None:
        self.taxons.append(taxon)

