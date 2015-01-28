#!/usr/bin/Rscript

source("pqplot.r")

usage <- function() {
	cat("Syntax: plot.r mode csv-files... classifiers... outfile",
	    "Modes: acc  Plot the accuracy and standard deviation for each classifier per dataset.",
	    "       atr  Plot the accuracy depending on the training time for each classifier.",
	    "       ats  Plot the accuracy depending on the testing time for each classifier.",
	    "       The modes atr and ats support only one dataset.",
	    "plot.r creates the specified plot with the data from the csv files and stores the",
	    "result in outfile as eps.", sep="\n")

	quit(save = "no", status = 1, runLast = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
	usage()
}

mode <- args[1]
outfile <- args[length(args)]
args <- args[2:(length(args)-1)]

csv_files <- args[grepl(".*\\.csv$", args)]
classifiers <- setdiff(args, csv_files)

print(mode)
print(csv_files)
print(classifiers)
print(outfile)

if (mode == "acc") {
	plot_accuracy(csv_files, classifiers, outfile)
} else if (mode == "atr") {
	plot_accuracy_by_train_time(csv_files, classifiers, outfile)
} else if (mode == "ats") {
	plot_accuracy_by_test_time(csv_files, classifiers, outfile)
} else {
	usage()
}
