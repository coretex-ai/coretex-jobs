from pathlib import Path

from coretex import CustomSample
from coretex.bioinformatics import ctx_qiime2


def summarizeSample(sample: CustomSample, outputDir: Path) -> Path:
    demuxPath = Path(sample.path) / "demux.qza"
    visualizationPath = outputDir / "demux.qzv"

    ctx_qiime2.demuxSummarize(str(demuxPath), str(visualizationPath))
    return visualizationPath
