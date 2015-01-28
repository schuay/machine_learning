library(Rmisc)
library(extrafont)
library(fontcm)
library(ggplot2)
library(plyr)
library(scales)

plot_accuracy <- function(file_names, classifiers, figfile=NULL) {
    if (is.null(figfile)) {
    	figbasename = strsplit(as.character(runif(1, 0, 1)), "\\.")[[1]][2]
    	figfile <- paste(figbasename, ".eps", sep = "")
    }

    data <- NULL
    for (fn in file_names) {
        f <- read.csv(fn)
        data <- rbind(data, f)
    }

    data <- subset(data, classifier %in% classifiers)
    data$classifier <- factor(data$classifier, levels = data$classifier, ordered = TRUE)

    p <- ggplot(data, aes(x=dataset, y=mean_accuracy, fill=classifier)) +
            geom_bar(position=position_dodge(), stat="identity") +
            geom_errorbar(aes(ymin=mean_accuracy-std_deviation, ymax=mean_accuracy+std_deviation),
                    width=.2, position=position_dodge(.9)) +
            ylab("Accuracy") +
            coord_cartesian(ylim = c(0.5, 1.0)) +
            theme_bw() +
                theme(text = element_text(size = 20),
                      axis.text = element_text(size = 18),
                      axis.title = element_text(size = 20),
                      legend.text = element_text(size = 18),
                      legend.title = element_text(size = 20))

    postscript(file = figfile,
               paper = "special",
               width = 10,
               height = 5,
               horizontal = FALSE,
               family = "CM Roman")
    plot(p)
    invisible(dev.off())
    cat("\\includegraphics[width = \\textwidth]{", figfile, "}\n\n", sep = "")
}

plot_accuracy_by_train_time <- function(file_names, classifiers, figfile=NULL) {
    if (is.null(figfile)) {
    	figbasename = strsplit(as.character(runif(1, 0, 1)), "\\.")[[1]][2]
    	figfile <- paste(figbasename, ".eps", sep = "")
    }

    data <- NULL
    for (fn in file_names) {
        f <- read.csv(fn)
        data <- rbind(data, f)
    }

    data <- subset(data, classifier %in% classifiers)

    p <- ggplot(data, aes(x=mean_train_time, y=mean_accuracy, colour=classifier, shape=classifier)) +
            scale_shape_manual(values=1:nlevels(data$classifier)) +
            geom_point(size=3) +
            ylab("Accuracy") +
            xlab("Train time (s)") +
            theme_bw() +
                theme(text = element_text(size = 20),
                      axis.text = element_text(size = 18),
                      axis.title = element_text(size = 20),
                      legend.text = element_text(size = 18),
                      legend.title = element_text(size = 20))

    postscript(file = figfile,
               paper = "special",
               width = 10,
               height = 5,
               horizontal = FALSE,
               family = "CM Roman")
    plot(p)
    invisible(dev.off())
    cat("\\includegraphics[width = \\textwidth]{", figfile, "}\n\n", sep = "")
}

plot_accuracy_by_test_time <- function(file_names, classifiers, figfile=NULL) {
    if (is.null(figfile)) {
    	figbasename = strsplit(as.character(runif(1, 0, 1)), "\\.")[[1]][2]
    	figfile <- paste(figbasename, ".eps", sep = "")
    }

    data <- NULL
    for (fn in file_names) {
        f <- read.csv(fn)
        data <- rbind(data, f)
    }

    data <- subset(data, classifier %in% classifiers)

    p <- ggplot(data, aes(x=mean_test_time, y=mean_accuracy, colour=classifier, shape=classifier)) +
            scale_shape_manual(values=1:nlevels(data$classifier)) +
            geom_point(size=3) +
            ylab("Accuracy") +
            xlab("Test time (s)") +
            theme_bw() +
                theme(text = element_text(size = 20),
                      axis.text = element_text(size = 18),
                      axis.title = element_text(size = 20),
                      legend.text = element_text(size = 18),
                      legend.title = element_text(size = 20))

    postscript(file = figfile,
               paper = "special",
               width = 10,
               height = 5,
               horizontal = FALSE,
               family = "CM Roman")
    plot(p)
    invisible(dev.off())
    cat("\\includegraphics[width = \\textwidth]{", figfile, "}\n\n", sep = "")
}
