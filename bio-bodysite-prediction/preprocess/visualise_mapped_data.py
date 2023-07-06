mappedFilePath = "" # path to atlas .mapped file

print(".MAPPED FILE: \n")
print("Header:")
print("Sample id, \t RNA read count, \t mapped rads, \t OTU count \n")

with open(mappedFilePath, "r") as mappedFile:
    for i in range(1, 100):
        print(mappedFile.readline())
