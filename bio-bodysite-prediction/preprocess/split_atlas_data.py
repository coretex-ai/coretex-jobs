#####################################################################
##### THIS IS DEPRECATED AFTER IMPLEMENTATION OF CHUNEKD UPLOAD #####
#####################################################################

# This script is used for processing one Microbe Atlas Dataset
# .mapped file and as output it generates couple of smaller json files

import os
import zipfile

from coretex import CustomSample
from coretex.utils import guessMimeType
from coretex.networking import NetworkManager
from coretex.threading import MultithreadedDataProcessor


NetworkManager.instance().authenticate(username = "", password = "") # pass your login credentials here username and password
filePath = "" # path to atlas .mapped file
infoPath = ""
listOfFiles = []


def createItem(path: str) -> CustomSample:
    mimeType = guessMimeType(path)

    files = [
            ("file", ("file", open(path, "rb"), mimeType))
        ]

    parameters = {
        "dataset_id": 1234,
        "name": os.path.basename(path),
    }

    response = NetworkManager.instance().genericUpload("session/import", files, parameters)
    if response.hasFailed():
        return "Failed importing this file!"


# Define the file path and the desired size of each split
split_size = 500 * 1024 * 1024  # 500 MB in bytes

# Open the input file and read its contents
with open(filePath, 'rb') as f:
    file_contents = f.read()

# Get the total size of the input file in bytes
file_size = os.path.getsize(filePath)

# Calculate the number of splits needed
num_splits = file_size // split_size + 1

# Write each split to a separate file
for i in range(num_splits):
    # Determine the start and end positions of the current split
    start_pos = i * split_size
    end_pos = min((i + 1) * split_size, file_size)

    # Extract the contents of the current split
    split_contents = file_contents[start_pos:end_pos]

    # Write the current split to a separate file
    output_path = f"part{i}.mapped"
    with open(output_path, 'wb') as f:
        f.write(split_contents)

    # Zip the current split into a separate archive
    zip_path = f"part{i}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        zip_file.write(output_path)

    # Remove the temporary split file
    os.remove(output_path)

for file in os.listdir(os.getcwd()):
    if file.endswith(".zip"):
        file = os.path.join(os.getcwd(), file)
        listOfFiles.append(file)

createItem(infoPath)

processor = MultithreadedDataProcessor(
    listOfFiles,
    createItem,
    threadCount = 8, #set threadCount on desired value default is 8
    title = "createItem"
)

processor.process()
