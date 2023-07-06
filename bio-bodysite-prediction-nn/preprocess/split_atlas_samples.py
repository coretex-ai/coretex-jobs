# This script is used for processing one Microbe Atlas Dataset (json) file and as output
# it generates 1 json file for every sample there is in that file.
# Only thing you have to do is to set up path to your Atlas dataset .mapped and .info file and run the script.
# This way when you upload dataset to Coretex you will have as many Coretex samples
# as you have samples in your .json file


from pathlib import Path
from zipfile import ZipFile

from src.utils import getBodySite, jsonPretty


mappedPath = "/Users/bogdanbm/Documents/samples-otus.90.mapped" # path to atlas .mapped file
sampleInfoPath = "/Users/bogdanbm/Documents/samples.env.info" # path to atlas samples.info file
datasetPath = Path("dataset1") # path to local folder where u want samples to be saved

sampleInfoObj: dict[str, str] = {}
sampleInfoList: list[dict] = []

uniqueBodySite: dict[str, int] = {}
mappedSampleObjList: list[dict] = []

with open(sampleInfoPath, "r") as sampleInfoFile:
    while True:
        line = sampleInfoFile.readline()

        if not line:
            break

        sampleInfoList.append(line.strip().split("\t"))

for info in sampleInfoList:
    if len(info) >= 2:
        if "human" in info[1]:
            sampleInfoObj[info[0]] = str(info[2]) + "," + info[4]

with open(mappedPath, "r") as mappedFile:
    saveState = False
    appendState = False

    taxonList: list[dict] = []
    uniqueTaxons: dict[str, int] = {}
    uniqueBodySite: dict[str, int] = {}
    taxonDistribution: dict[str, int] = {}

    while True:
        line = mappedFile.readline()

        if not line:
            break

        splitLine = line.strip().split("\t")

        if splitLine[0].startswith(">"):
            appendState = False

            if saveState:
                mappedSampleObj["taxons"] = taxonList

                jsonPretty(mappedSampleObj, datasetPath / f"sample{len(mappedSampleObjList)}.json")

                with ZipFile(datasetPath / f"sample{len(mappedSampleObjList)}.zip", mode = "w") as archive:
                    archive.write(datasetPath / f"sample{len(mappedSampleObjList)}.json", f"sample{len(mappedSampleObjList)}.json")

                Path(datasetPath / f"sample{len(mappedSampleObjList)}.json").unlink(missing_ok=True)

                mappedSampleObjList.append(mappedSampleObj)
                saveState = False

                taxonList = []

            if getBodySite(splitLine[0][1:], sampleInfoObj) is not None:
                body_site, association_site = getBodySite(splitLine[0][1:], sampleInfoObj)

                body_site = body_site.split(";")[0]
                if not body_site in uniqueBodySite:
                    uniqueBodySite[body_site] = len(uniqueBodySite)

                mappedSampleObj = {}
                mappedSampleObj["sample_id"] = splitLine[0][1:]
                mappedSampleObj["body_site"] = body_site
                mappedSampleObj["association_site"] = association_site

                appendState = True
                saveState = True
        else:
            if appendState:
                taxon = {}
                taxon["taxon"] = splitLine[0]

                if not splitLine[0] in uniqueTaxons:
                    uniqueTaxons[splitLine[0]] = len(uniqueTaxons)

                taxon["count"] = splitLine[1]
                taxonList.append(taxon)
