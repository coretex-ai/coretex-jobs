# This script is used for processing one Microbiome Forensics (json) file and as output
# it generates 1 json file for every sample there is in that file.
# Only thing you have to do is to set up path to your MBF .json file and run the script.
# This way when you upload dataset to Coretex you will have as many Coretex samples
# as you have samples in your .json file

from pathlib import Path
from zipfile import ZipFile

import os
import json


path = "" # write path to folder with json files

for file in os.listdir(path):
    if file.endswith(".json"):
        with open(os.path.join(path, file), "r") as f:
            data = json.load(f)

        for i, sample in enumerate(data):
            with open(f"{os.path.join(path, file[:-5])}{i}.json", "w") as sF:
                json.dump(sample, sF, indent=4)

            with ZipFile(f"{os.path.join(path, file[:-5])}{i}.zip", mode = "w") as archive:
                archive.write(f"{os.path.join(path, file[:-5])}{i}.json", f"{file[:-5]}{i}.json")

            Path(f"{os.path.join(path, file[:-5])}{i}.json").unlink(missing_ok=True)
