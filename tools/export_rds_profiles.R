#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || length(args) > 3L) {
  stop("Usage: export_rds_profiles.R INPUT.rds OUTPUT.tsv [MAX_SAMPLES]")
}

input_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_path <- args[[2L]]
max_samples <- if (length(args) == 3L) as.integer(args[[3L]]) else NA_integer_

data <- readRDS(input_path)
required_columns <- c("chromosome", "start", "end", "segVal", "sample")
if (!is.data.frame(data) || !all(required_columns %in% colnames(data))) {
  stop("Input must be a data.frame with columns: ",
       paste(required_columns, collapse = ", "))
}

sample_ids <- unique(as.character(data$sample))
if (!is.na(max_samples)) {
  sample_ids <- head(sample_ids, max_samples)
}
data <- data[data$sample %in% sample_ids, required_columns, drop = FALSE]
data$sample <- factor(data$sample, levels = sample_ids)
data <- data[order(data$sample, as.integer(data$chromosome), data$start), ]

sample_rows <- split(data, data$sample, drop = TRUE)
segment_keys <- lapply(sample_rows, function(frame) {
  paste(frame$chromosome, frame$start, frame$end, sep = ":")
})
if (!all(vapply(segment_keys[-1L], identical, logical(1L), segment_keys[[1L]]))) {
  stop("Selected samples do not contain identical ordered 100 KB bins")
}

encode_profile <- function(frame) {
  chromosomes <- split(
    as.character(as.integer(frame$segVal)),
    factor(frame$chromosome, levels = unique(frame$chromosome)))
  paste(vapply(chromosomes, paste0, character(1L), collapse = ""),
        collapse = "X")
}

profiles <- vapply(sample_rows, encode_profile, character(1L))
diploid <- gsub("[0-9]", "2", profiles[[1L]])
output <- data.frame(
  sample_id = c(names(profiles), "diploid"),
  profile = c(unname(profiles), diploid),
  stringsAsFactors = FALSE)

dir.create(dirname(normalizePath(output_path, mustWork = FALSE)),
           recursive = TRUE, showWarnings = FALSE)
write.table(output, output_path, sep = "\t", quote = FALSE,
            row.names = FALSE, col.names = TRUE)

bin_widths <- data$end - data$start + 1L
cat("input=", input_path, "\n", sep = "")
cat("output=", normalizePath(output_path, mustWork = TRUE), "\n", sep = "")
cat("cells=", length(profiles), "\n", sep = "")
cat("bins_per_cell=", nrow(sample_rows[[1L]]), "\n", sep = "")
cat("chromosomes=", length(unique(sample_rows[[1L]]$chromosome)), "\n", sep = "")
cat("median_bin_width=", stats::median(bin_widths), "\n", sep = "")
