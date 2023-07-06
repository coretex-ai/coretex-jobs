infoFilePath = ""

with open(infoFilePath, "r") as infoFile:
    for i in range(1, 100):
        print(infoFile.readline())