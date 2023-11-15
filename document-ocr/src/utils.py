import logging

import matplotlib.pyplot as plt

from coretex import folder_manager


def savePlot(name, image, mask, taskRun) -> None:
    plotPath = folder_manager.temp / f"{name}.png"
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(image)
    axes[1].imshow(mask)

    plt.savefig(plotPath)
    plt.close()

    if taskRun.createArtifact(plotPath, plotPath.name) is None:
        logging.warning("\tFailed to upload prediction as artifact")

    plotPath.unlink()
