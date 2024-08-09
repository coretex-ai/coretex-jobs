from typing import Optional


class Taxon:

    def __init__(self, taxonId: str, count: int):
        self.taxonId = taxonId
        self.count = count


class Sample:

    def __init__(self, sampleId: str, bodySite: str, associationSite: Optional[str], taxons: Optional[list[Taxon]] = None) -> None:
        self.sampleId = sampleId
        self.bodySite = bodySite
        self.associationSite = associationSite
        self.taxons = [] if taxons is None else taxons

    def addTaxon(self, taxon: Taxon) -> None:
        if taxon not in self.taxons:
            self.taxons.append(taxon)
        else:
            raise RuntimeError(f">> [MicrobiomeForensics] Taxon {taxon.taxonId} already in sample {self.sampleId}")
