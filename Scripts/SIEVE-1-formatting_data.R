# Validation of prediction models
# Objective : Compare observed and predicted values in RNA abundance
# Created by: Guillaume Ramstein (ramstein@qgg.au.dk)
# Created on: 7/2/2025

#--------------------------------------------------------
# Script parameters
#--------------------------------------------------------
# Working directory
setwd("~/sieve/WP2/")

# PCs
max_PCs <- 20

# Included genes
liftover_file <- "INPUT/model_validation/sieve/gene.id.translation.tsv"

# RNA-seq data
RNA_dir <- "INPUT/model_validation/sieve/RNAseq/"

transcript_file <- paste0(RNA_dir, "BdistachyonBd21_3_537_v1.2.protein_primaryTranscriptOnly.fa.gz")
ID2line_file <- paste0(RNA_dir, "RNAseq.ids.tsv")
RNA_file <- paste0(RNA_dir, "RNAseq.reads.tsv")
QC_dirs <- paste0(RNA_dir,
                  c(
                    "90-1106610255/01_output/",
                    "90-1120364334/01_output/",
                    "90-1080311147/01_output/",
                    "90-1147085216/01_output/"
                  )
)

QC_threshold <- 70

# Model predictions
phytoExpr_dirs <- c(
  "phytoExpr_B"="OUTPUT/sieve/phytoexpr_predictions/predictions_B",
  "phytoExpr_C"="OUTPUT/sieve/phytoexpr_predictions/predictions_C",
  "phytoExpr_B_July2026"="/home/ramstein/sieve/WP2/DATA/archives/PhytoExpr/sieve_population/output/pred_out.bdi_B",
  "phytoExpr_C_July2026"="/home/ramstein/sieve/WP2/DATA/archives/PhytoExpr/sieve_population/output/pred_out.bdi_C"
)

empres_files <- c(
  "EMPRES_0"="OUTPUT/sieve/predictions_EMPRES_0.tsv",
  "EMPRES_1"="OUTPUT/sieve/predictions_BD_none.tsv",
  "EMPRES_2"="OUTPUT/sieve/predictions_BD_pred.tsv",
  "EMPRES_3"="OUTPUT/sieve/predictions_BD_emb.tsv",
  "EMPRES_4"="OUTPUT/sieve/predictions_BD_a2z.tsv",
  "EMPRES_0_July2026"="OUTPUT/sieve/July2026/predictions_EMPRES_0.tsv"
)

# Output
output_dir <- "OUTPUT/sieve/validation/"
dir.create(output_dir, showWarnings=FALSE)

formatted_file <- paste0(output_dir, "SIEVE_M6_TPM.csv")
controls_file <- paste0(output_dir, "between_gene_variation.rds")
mutants_file <- paste0(output_dir, "within_gene_variation.rds")

#--------------------------------------------------------
# Functions
#--------------------------------------------------------
library(tidyverse)
library(data.table)
library(foreach)

library(ggplot2)

library(MASS)

library(broom)
library(tidyr)

library(sensemakr)
library(arm)

capitalize <- function(x) {
  paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x)))
}

#--------------------------------------------------------
# Data
#--------------------------------------------------------
# Line information
id2line <- read.table(ID2line_file, header=TRUE, sep='\t', fill=TRUE, stringsAsFactors=FALSE)

# All transcripts
transcripts <- readLines(transcript_file, warn=FALSE) %>%
  grep("^>", ., value=TRUE) %>%
  str_match("(?:^|\\s)transcript=(\\S+)") %>%
  .[, 2]

genes <- readLines(transcript_file, warn=FALSE) %>%
  grep("^>", ., value=TRUE) %>%
  str_match("(?:^|\\s)locus=(\\S+)") %>%
  .[, 2]

stopifnot(length(genes) == length(transcripts),
          !anyNA(genes),
          !anyNA(transcripts))

# Included transcripts
included_transcripts <- read.table(liftover_file, header=FALSE)[, 1]

idx <- which(transcripts %in% included_transcripts)
transcripts <- transcripts[idx]
genes <- genes[idx]

names(genes) <- transcripts

# Expression data
expression <- fread(RNA_file, header=TRUE) %>%
  column_to_rownames("gene") %>%
  as.matrix() %>%
  t()

id <- as.integer(sub("_TPM", "", rownames(expression)))
rownames(expression) <- id2line[match(id, id2line$ID), "line"]

colnames(expression) <- sub("[.]v1[.]2$", "", colnames(expression))

obs <- reshape2::melt(expression,
                 varnames=c("id", "gene"),
                 value.name="tpm") %>%
  dplyr::mutate(log_tpm=log10(1+tpm)) %>%
  dplyr::mutate_if(is.factor, as.character)

# Output
formatted <- cbind(id=rownames(expression), data.table(expression))
fwrite(formatted, formatted_file)

#--------------------------------------------------------
# Predictions
#--------------------------------------------------------
# PhytoExpr
pred <- foreach(dir_name=names(phytoExpr_dirs), .combine=merge) %do% {
  
  phytoExpr_dir <- phytoExpr_dirs[dir_name]
  
  files <- list.files(phytoExpr_dir, full.names=TRUE)
  
  out <- foreach(file_name=files, .combine=rbind) %do% {
    
    fread(file_name) %>%
      dplyr::rename(id=group_for_cross_validation) %>%
      dplyr::select(id, transcript, Pred_median_TPM)
    
  }
  
  names(out)[names(out) == "Pred_median_TPM"] <- dir_name
  
  stopifnot(!anyDuplicated(out[, c("id", "transcript")]))
  
  return(out)
  
} %>%
  dplyr::filter(transcript %in% .GlobalEnv$transcripts)

pred <- dplyr::mutate(pred, id=strsplit(id, " ")) %>%
  unnest(id) %>% 
  dplyr::mutate(case=paste(transcript, id, sep="_")) %>%
  as.data.table()

# EMPRES
for (empres_name in names(empres_files)) {
  
  print(empres_name)
  
  # Observed and prediction expression values
  DF <- fread(empres_files[empres_name]) %>%
    dplyr::filter(transcript %in% .GlobalEnv$transcripts) %>%
    as.data.frame()
  
  # Averaging predictions
  names(DF) <- sub("_pred.*", "", names(DF))
  DF[, empres_name] <- rowMeans(DF[, paste0("model_", 1:5), drop=FALSE])
  
  # Output
  pred <- DF[, c("id", "transcript", empres_name)] %>%
    as.data.table() %>%
    merge(pred, by=c("id", "transcript"))
  
  stopifnot(!anyDuplicated(pred[, c("id", "transcript")]))
  message(empres_name, ": ", nrow(pred), " rows")
  
}

rm(DF)
gc()

# Filtering based on QC
QC <- foreach(QC_dir=QC_dirs, .combine=rbind) %do% {
  
  file_names <- list.files(QC_dir, full.names=TRUE)
  file_names <- file_names[grep("_summary.txt", file_names)]
  
  foreach(file_name=file_names, .combine=rbind) %do% {
    
    txt <- scan(file_name, what="", sep="\n", quiet=TRUE)
    txt <- gsub("\t", "", txt)
    
    stat <- txt[grep("Aligned concordantly 1 time", txt)]
    stat <- sub("%[)]$", "", sub("^.+[(]", "", stat))
    
    data.frame(line=sub("_.+$", "", sub("^.+[/]", "", file_name)),
               stat=as.numeric(stat)
    )
    
  }
  
}

selected_lines <- QC$line[which(QC$stat >= QC_threshold)]

# PCA
selected_samples <- id2line$line[match(selected_lines, id2line$ID)]

X <- log10(1+expression[selected_samples, ])

PCA <- prcomp(X[, colSums(X) > 0, drop=FALSE], scale.=TRUE, rank.=max_PCs)

DF_PCA <- data.frame('id'=rownames(PCA$x), PCA$x, stringsAsFactors=FALSE)

# Merging with observed values
pred$line <- id2line$ID[match(pred$id, id2line$line)]
pred <- pred[pred$line %in% selected_lines, ]

pred$gene <- genes[pred$transcript]

pred <- merge(pred, as.data.table(obs), by=c("id", "gene")) %>%
  as.data.frame()

#--------------------------------------------------------
# Output
#--------------------------------------------------------
# Averages among controls
xvars <- c(names(phytoExpr_dirs), names(empres_files))
yvar <- "log_tpm"

controls <- pred[startsWith(pred$id, "C"), ]
controls <- aggregate(controls[, c(xvars, yvar)], list('gene'=controls$gene), mean)

saveRDS(controls, controls_file)

# Predictions from controls and mutants
mutants <- data.table(pred[startsWith(pred$id, "M"), ]) %>%
  merge(data.table(controls), by="gene", suffixes=c(".mutant", ".control")) %>%
  dplyr::mutate(delta.obs=log_tpm.mutant-log_tpm.control) %>%
  as.data.frame() %>%
  merge(DF_PCA, by="id")

saveRDS(mutants, mutants_file)
