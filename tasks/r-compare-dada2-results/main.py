import csv
import logging

from coretex import currentTaskRun, Artifact


def isTrackFileArtifact(artifact: Artifact) -> bool:
    if not artifact.remoteFilePath.startswith("track_trimLeft_trimRight_"):
        return False

    if not artifact.remoteFilePath.endswith(".csv"):
        return False

    return True


def getDada2Result(artifactId: int) -> float:
    artifacts = Artifact.fetchAll(artifactId)
    filtered = [artifact for artifact in artifacts if isTrackFileArtifact(artifact)]

    if len(filtered) > 1:
        raise RuntimeError(f"Found more than one track file for Run: {artifactId}")

    if len(filtered) == 0:
        raise RuntimeError(f"Failed to find track file for Run: {artifactId}")

    artifact = filtered[0]
    artifact.localFilePath.parent.mkdir(parents = True, exist_ok = True)
    artifact.download()

    with artifact.localFilePath.open("r") as file:
        data: list[float] = []

        for row in csv.DictReader(file):
            data.append(float(row["nonchimPct"]))

        if len(data) == 0:
            return 0

        return sum(data) / len(data)


def main() -> None:
    taskRun = currentTaskRun()

    logging.info(">> [Coretex] Comparing results...")
    results: dict[int, float] = {}

    for artifactId in taskRun.parameters["dada2Artifacts"]:
        result = getDada2Result(artifactId)
        results[artifactId] = result

        logging.info(f"\tRun: {artifactId} has quality: {result}")

    highest = max(results, key = lambda key: results[key])
    logging.info(f">> [Coretex] Run with ID: {highest} has highest quality")

    taskRun.submitOutput("artifacts", highest)


if __name__ == "__main__":
    main()
