args <- commandArgs(trailingOnly = TRUE)

library(reticulate)

os <- import("os")
ctx <- import("coretex")
ctxFolderManager <- import("coretex.folder_manager")

main <- function(taskRun) {
    # FolderManager.temp is used for temporary file storage
    # temp directory is cleared when the TaskRun execution is finished
    plotPath <- os$path$join(ctxFolderManager$temp, "plot.png")

    # TaskRun.parameters contain parameters entered when the TaskRun
    # was created on Coretex.ai
    mean <- taskRun$parameters[["mean"]]
    sd <- taskRun$parameters[["sd"]]

    print(">> [R Example] Plotting normal distribution...")
    print(sprintf(">> [R Example] Mean value: %.2f, Standard deviation value: %.2f", mean, sd))

    png(plotPath)

    # Create a sequence of 100 equally spaced numbers between -4 and 4
    x <- seq(-4, 4, length = 100)

    # Create a vector of values that shows the height of the
    # probability distribution for each value in x
    y <- dnorm(x, mean, sd)

    # Plot x and y as a scatterplot with connected lines (type = "l")
    plot(x, y, type = "l", lwd = 2, axes = FALSE, xlab = "", ylab = "")
    dev.off()

    # TaskRun.createArtifact is used to store any kind of file to
    # Coretex.ai TaskRun artifacts
    if (is.null(taskRun$createArtifact(plotPath, "plot.png"))) {
        print(">> [R Example] Failed to create Artifact with name \"plot.png\"")
    } else {
        print(">> [R Example] Artifact with name \"plot.png\" created")
    }
}

# initializeTask must be called with a function to start the TaskRun execution
ctx$initializeRTask(main, args)
