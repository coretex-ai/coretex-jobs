from pathlib import Path

import subprocess
import logging

from .filepaths import BWA, SAMTOOLS


def logProcessOutput(output: bytes, severity: int) -> None:
    decoded = output.decode("UTF-8")

    for line in decoded.split("\n"):
        # skip empty lines
        if line.strip() == "":
            continue

        # ignoring type for now, has to be fixed in coretexpylib
        logging.getLogger("sequenceAlignment").log(severity, line)


def command(args: list[str]) -> None:
    process = subprocess.Popen(
        args,
        shell = False,
        cwd = Path(__file__).parent,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )

    while process.poll() is None:
        stdout = process.stdout.readline()
        stderr = process.stderr.readline()

        if len(stdout) > 0:
            logProcessOutput(stdout, logging.INFO)

        if len(stderr) > 0:
            if process.returncode == 0:
                logProcessOutput(stdout, logging.INFO)
            else:
                logProcessOutput(stderr, logging.CRITICAL)

    if process.returncode != 0:
        raise RuntimeError(f">> [Sequence Alignment] Falied to execute command. Returncode: {process.returncode}")


def indexCommand(genomePath: Path, genomePrefix: Path) -> None:
    bwaPath = Path(BWA)
    command([
        str(bwaPath.absolute()), "index",
        "-p", str(genomePrefix.absolute()),
        str(genomePath.absolute())
    ])


def alignCommand(genomePrefix: Path, sequencePath: Path, outputPath: Path) -> None:
    bwaPath = Path(BWA)
    command([
        str(bwaPath.absolute()), "mem",
        "-o", str(outputPath.absolute()),
        str(genomePrefix.absolute()),
        str(sequencePath.absolute())
    ])


def sam2bamCommand(samPath: Path, outputPath: Path) -> None:
    samtoolsPath = Path(SAMTOOLS)
    command([
        str(samtoolsPath.absolute()), "view",
        "-b", "-S", "-o",
        str(outputPath.absolute()),
        str(samPath.absolute())
    ])


def chmodX(file: Path) -> None:
    command(["chmod", "+x", str(file.absolute())])
