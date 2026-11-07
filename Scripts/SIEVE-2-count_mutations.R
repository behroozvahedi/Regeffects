# Mutation counts by type and genotype state
# Objective: Count mutations by transcript, id, and state

#--------------------------------------------------------
# Script parameters
#--------------------------------------------------------
setwd("~/sieve/WP2/")

# Files
input_file <- "INPUT/model_validation/sieve/data.bd.csv"
output_file <- "OUTPUT/sieve/mutation_counts.tsv"

# Optional list of transcripts to keep
primary_file <- NULL

# Write all-zero rows for the lines carrying the most common sequence
include_reference_ids <- FALSE

# Expected length of each promoter and terminator sequence
seq_length <- 10000

# Windows compared
tss_window <- c(1001, 6000)   # 4,000 bp upstream + 1,000 bp downstream of the TSS
tts_window <- c(4001, 9000)   # 1,000 bp upstream + 4,000 bp downstream of the TTS

#--------------------------------------------------------
# Functions
#--------------------------------------------------------
library(data.table)

bases <- c("A", "T", "C", "G")

# Mutation types
mutation_types <- unlist(lapply(bases, function(b) paste0(b, ">", setdiff(bases, b))))

het_codes <- list(W=c("A", "T"), S=c("C", "G"), M=c("A", "C"),
                  K=c("G", "T"), R=c("A", "G"), Y=c("C", "T"))

het_types <- unname(unlist(lapply(bases, function(b) {                            
  sapply(setdiff(bases, b), function(m) {                                         
    code <- names(het_codes)[sapply(het_codes, function(x) setequal(x, c(b, m)))] 
    paste0(b, ">", code)                                                          
  })                                                                              
})))                                                                              

fixed_types <- c(mutation_types, het_types)

# Positions where two sequences of equal length differ
compare_sequences <- function(ref_raw, alt) {

  alt_raw <- charToRaw(alt)
  d <- which(ref_raw != alt_raw)

  if (length(d) == 0) return(character(0))

  paste0(rawToChar(ref_raw[d], multiple=TRUE), ">",
         rawToChar(alt_raw[d], multiple=TRUE))         

}

# Counts per mutation type
count_types <- function(types, levels) {              
  as.integer(table(factor(types, levels=levels)))
}

# Transcript IDs from a FASTA file
read_transcripts <- function(file) {

  lines <- readLines(file, warn=FALSE)

  if (any(startsWith(lines, ">"))) {
    headers <- grep("^>", lines, value=TRUE)
    ids <- regmatches(headers, regexpr("(?<=transcript=)\\S+", headers, perl=TRUE))
    if (length(ids) != length(headers)) {
      stop("Some FASTA headers in ", file, " have no 'transcript=' field.", call.=FALSE)
    }
  } else {
    ids <- sub("[[:space:],].*$", "", trimws(lines))
    ids <- ids[ids != "" & tolower(ids) != "transcript"]
  }

  unique(ids)

}

#--------------------------------------------------------
# Input
#--------------------------------------------------------
# Sequence file
required <- c("gene", "transcript", "promoter", "terminator", "group_for_cross_validation")

header <- gsub('^"|"$', "", strsplit(readLines(input_file, n=1), ",")[[1]])
missing_cols <- setdiff(required, header)
if (length(missing_cols) > 0) {
  stop("Input is missing column(s): ", paste(missing_cols, collapse=", "), call.=FALSE)
}

DF <- as.data.frame(fread(input_file, select=required, colClasses="character",
                          na.strings=c("", "NA")))
cat("Read", nrow(DF), "sequences from", input_file, "\n")

# Transcript filter
if (!is.null(primary_file)) {
  keep <- read_transcripts(primary_file)
  n_before <- nrow(DF)
  DF <- DF[DF$transcript %in% keep, , drop=FALSE]
  cat("Kept", nrow(DF), "of", n_before, "sequences on transcripts listed in", primary_file, "\n")
}

# Rows without sequences
no_seq <- is.na(DF$promoter) | is.na(DF$terminator)
if (any(no_seq)) {
  warning(sum(no_seq), " rows without a promoter or terminator sequence were dropped.", call.=FALSE)
  DF <- DF[!no_seq, , drop=FALSE]
}

# Length check
bad_length <- nchar(DF$promoter) != seq_length | nchar(DF$terminator) != seq_length

if (any(bad_length)) {
  stop(sum(bad_length), " sequences are not ", seq_length, " bp long (first one on transcript ",
       DF$transcript[which(bad_length)[1]], ")",
       call.=FALSE)
}

# Windows compared
DF$tss <- toupper(substr(DF$promoter, tss_window[1], tss_window[2]))
DF$tts <- toupper(substr(DF$terminator, tts_window[1], tts_window[2]))
DF$promoter <- NULL
DF$terminator <- NULL

# Lines carrying each sequence
ids_list <- lapply(strsplit(DF$group_for_cross_validation, "[[:space:],;]+"), function(x) {
  unique(x[!is.na(x) & x != "" & toupper(x) != "NA"])
})

#--------------------------------------------------------
# Counting mutations
#--------------------------------------------------------
# Comparison groups: gene + transcript, in order of first appearance
group_key <- paste(DF$gene, DF$transcript, sep="\r")
groups <- split(seq_len(nrow(DF)), factor(group_key, levels=unique(group_key)))

# One record per sequence written (alternate sequences, and optionally the reference)
rec_row <- integer(nrow(DF))
rec_ids <- vector("list", nrow(DF))
rec_types <- vector("list", nrow(DF))
k <- 0

n_ties <- 0               # groups where several sequences share the largest number of lines
n_no_ids <- 0             # alternate sequences carried by no listed line

for (idx in groups) {

  # Most common sequence: carried by the most lines (first one in case of a tie)
  n_ids <- lengths(ids_list[idx])
  ref_pos <- which.max(n_ids)
  if (sum(n_ids == max(n_ids)) > 1) n_ties <- n_ties + 1

  ref <- idx[ref_pos]
  ref_tss <- charToRaw(DF$tss[ref])
  ref_tts <- charToRaw(DF$tts[ref])

  for (alt in idx[-ref_pos]) {

    ids <- ids_list[[alt]]
    if (length(ids) == 0) {
      n_no_ids <- n_no_ids + 1
      next
    }

    types <- c(compare_sequences(ref_tss, DF$tss[alt]),   
               compare_sequences(ref_tts, DF$tts[alt])) 

    k <- k + 1
    rec_row[k] <- alt
    rec_ids[[k]] <- ids
    rec_types[[k]] <- types 

  }

  if (include_reference_ids && length(ids_list[[ref]]) > 0) {
    k <- k + 1
    rec_row[k] <- ref
    rec_ids[[k]] <- ids_list[[ref]]
    rec_types[[k]] <- character(0)
  }

}

#--------------------------------------------------------
# Output
#--------------------------------------------------------
observed <- as.character(unlist(rec_types[seq_len(k)]))                 
all_types <- c(fixed_types, sort(setdiff(unique(observed), fixed_types))) 

# One row per line: repeat each sequence's counts for every line carrying it
n_each <- lengths(rec_ids[seq_len(k)])
expand <- rep(seq_len(k), n_each)

meta <- data.frame(gene=DF$gene[rec_row[expand]],
                   transcript=DF$transcript[rec_row[expand]],
                   id=as.character(unlist(rec_ids[seq_len(k)])),
                   stringsAsFactors=FALSE)

if (k > 0) {
  count_mat <- do.call(rbind, lapply(rec_types[seq_len(k)], count_types,          
                                     levels=all_types))[expand, , drop=FALSE]     
} else {
  count_mat <- matrix(integer(0), nrow=0, ncol=length(all_types))                 
}
colnames(count_mat) <- all_types

# A line listed on several sequences of the same transcript: sum its counts
row_key <- paste(meta$gene, meta$transcript, meta$id, sep="\r")
if (anyDuplicated(row_key)) {
  cat("Summed counts for", sum(duplicated(row_key)), "repeated gene + transcript + line rows\n")
  count_mat <- rowsum(count_mat, row_key, reorder=FALSE)
  meta <- meta[!duplicated(row_key), , drop=FALSE]
}

make_table <- function(meta, mat) {
  out <- cbind(as.data.table(meta), as.data.table(mat))
  setnames(out, c("gene", "transcript", "id", all_types))
  out
}

fwrite(make_table(meta, count_mat), output_file, sep="\t")

# Summary
other_types <- setdiff(all_types, fixed_types)   

cat("Transcripts compared:", length(groups), "\n")
cat("Rows written (gene + transcript + line):", nrow(meta), "\n")
cat("Homozygous mutations (e.g. C>T), summed over rows:", sum(count_mat[, mutation_types]), "\n")   
cat("Heterozygous mutations (e.g. C>Y), summed over rows:", sum(count_mat[, het_types]), "\n")    
if (length(other_types) > 0) {                                                                    
  cat("Other differences, summed over rows:", sum(count_mat[, other_types, drop=FALSE]),          
      "in columns", paste(other_types, collapse=", "), "\n")                                     
}                                                                                                 
if (n_ties > 0) {
  cat("Transcripts where several sequences tie for most common (first one used):", n_ties, "\n")
}
if (n_no_ids > 0) {
  cat("Sequences skipped because no line carries them:", n_no_ids, "\n")
}
cat("Wrote", output_file, "\n")   
