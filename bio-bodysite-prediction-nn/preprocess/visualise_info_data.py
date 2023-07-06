infoFilePath = "/Users/bogdanbm/Documents/working/biomech/source/experiment-templates/bio-bodysite-prediction/dataForUpload/samples.env.info"

with open(infoFilePath, "r") as infoFile:
    for i in range(1, 100):
        print(infoFile.readline())