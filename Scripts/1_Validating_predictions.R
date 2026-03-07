# Validation of prediction models
# Objective : Compare observed and predicted values in RNA abundance
# Created by: Guillaume Ramstein (ramstein@qgg.au.dk)
# Created on: 7/2/2025

#--------------------------------------------------------
# Script parameters
#--------------------------------------------------------
# Working directory
setwd("~/sieve/WP2/DATA/PERL/sieve_population/output/")

# PCs
max_PCs <- 20

# PEER factors
n_PEERs <- 10

PEER_file <- paste0("PEER/peer_n-", n_PEERs, "/X.csv")

sample_file <- "peer.samples.csv"
gene_file <- "peer.genes.csv"
expression_file <- "peer.expression.csv"

# Model predictions
phytoExpr_dirs <- c(
  "phytoExpr_B"="~/sieve/WP2/DATA/PhytoExpr/sieve_population/output/pred_out.bdi_B/",
  "phytoExpr_C"="~/sieve/WP2/DATA/PhytoExpr/sieve_population/output/pred_out.bdi_C/"
)

empres_files <- c(
  "EMPRES_1"="predictions_none.tsv",
  "EMPRES_2"="predictions_pred.tsv",
  "EMPRES_3"="predictions_emb.tsv",
  "EMPRES_4"="predictions_a2z.tsv"
)

# Primary transcripts
transcript_file <- "BdistachyonBd21_3_537_v1.2.protein_primaryTranscriptOnly.tsv"

# ID to line number
ID2line_file <- "RNAseq.ids.tsv"

# QC
QC_dirs <- paste0("~/sieve/WP2/DATA/",
                   c(
                     "90-1106610255/01_output/",
                     "90-1120364334/01_output/",
                     "90-1080311147/01_output/",
                     "90-1147085216/01_output/"
                   )
)

QC_threshold <- 70

# Output
formatted_file <- "SIEVE_M6_TPM.csv"

output_file <- "regression_results.csv"
plot_file <- "plot_data.rds"
label_file <- "label_data.rds"

fig_dir <- "~/sieve/WP2/DATA/figs/"
dir.create(fig_dir, showWarnings=FALSE)

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

# Expression data
expression <- fread(expression_file, header=FALSE) %>%
  as.data.frame()

genes <- read.csv(gene_file, header=FALSE)$V1
samples <- read.csv(sample_file, header=FALSE)$V1
colnames(expression) <- genes
rownames(expression) <- samples

obs <- as.matrix(expression) %>%
  reshape2::melt(varnames=c("id", "gene"), value.name="tpm") %>%
  dplyr::mutate(log_tpm=log10(1+tpm)) %>%
  dplyr::mutate_if(is.factor, as.character)

# PEER factors
PEER_factors <- read.csv(PEER_file, header=FALSE)
PEER_samples <- read.table(sample_file, header=FALSE, sep='\t', stringsAsFactors=FALSE)

DF_PEER <- data.frame('id'= PEER_samples$V1,
                      t(PEER_factors),
                      stringsAsFactors=FALSE)

# PCA
PCA <- prcomp(log10(1+expression[, colSums(expression) != 0, drop=FALSE]),
              scale.=TRUE,
              rank.=max_PCs)

eigenvalues <- PCA$sdev^2

plot(eigenvalues/sum(eigenvalues),
     type="b",
     xlab="Principal component",
     ylab="Proportion of variance explained")

DF_PCA <- data.frame('id'=rownames(PCA$x), PCA$x, stringsAsFactors=FALSE)

# Primary transcripts
primary_transcripts <- read.table(transcript_file,
                                  header=TRUE,
                                  sep='\t',
                                  fill=TRUE,
                                  stringsAsFactors=FALSE)$transcript

# Output
formatted <- cbind(id=rownames(expression), expression)
colnames(formatted) <- c("id", paste0("BdiBd21-3.", colnames(expression)))

fwrite(formatted, formatted_file)

#--------------------------------------------------------
# Predictions
#--------------------------------------------------------
# PhytoExpr
pred <- foreach(dn=names(phytoExpr_dirs), .combine=merge) %do% {
  
  phytoExpr_dir <- phytoExpr_dirs[dn]
  
  files <- list.files(phytoExpr_dir, full.names=TRUE)
  
  out <- foreach(fn=files, .combine=rbind) %do% {
    
    fread(fn)
    
  }
  
  names(out)[names(out) == "Pred_median_TPM"] <- dn
  
  return(out)
  
}%>%
  dplyr::rename(id=group_for_cross_validation) %>%
  dplyr::filter(transcript %in% .GlobalEnv$primary_transcripts)

unique_cases <- with(pred[! grepl(" ", pred$id), ], paste(transcript, id, sep="_"))

pred <- dplyr::mutate(pred, id=strsplit(id, " ")) %>%
  unnest(id) %>% 
  dplyr::mutate(case=paste(transcript, id, sep="_")) %>%
  dplyr::select(transcript, id, case, phytoExpr_B, phytoExpr_C) %>%
  as.data.table()

# EMPRES
for (empres_name in names(empres_files)) {
  
  print(empres_name)
  
  # Observed and prediction expression values
  DF <- fread(empres_files[empres_name]) %>%
    dplyr::filter(transcript %in% .GlobalEnv$primary_transcripts) %>%
    as.data.frame()
  
  # Averaging predictions
  names(DF) <- sub("_pred.*", "", names(DF))
  DF[, empres_name] <- rowMeans(DF[, paste0("model_", 1:5), drop=FALSE])
  
  # Output
  pred <- DF[, c("id", "transcript", empres_name)] %>%
    as.data.table() %>%
    merge(pred, by=c("id", "transcript"))
  
}

# Filtering based on QC
pred$line <- id2line$ID[match(pred$id, id2line$line)]

QC <- foreach(QC_dir=QC_dirs, .combine=rbind) %do% {
  
  file_names <- list.files(QC_dir, full.names=TRUE)
  file_names <- file_names[grep("_summary.txt", file_names)]
  
  foreach(fn=file_names, .combine=rbind) %do% {
    
    txt <- scan(fn, what="", sep="\n", quiet=TRUE)
    txt <- gsub("\t", "", txt)
    
    stat <- txt[grep("Aligned concordantly 1 time", txt)]
    stat <- sub("%[)]$", "", sub("^.+[(]", "", stat))
    
    data.frame(line=sub("_.+$", "", sub("^.+[/]", "", fn)),
               stat=as.numeric(stat)
    )
    
  }
  
}

selected_lines <- QC$line[which(QC$stat >= QC_threshold)]

pred <- pred[pred$line %in% selected_lines, ]

# Merging with observed values
pred$gene <- sub("[.].+$", "", sub("^BdiBd21-3[.]", "", pred$transcript))
pred <- merge(pred, as.data.table(obs), by=c("id", "gene")) %>%
  as.data.frame()

#--------------------------------------------------------
# Analysis
#--------------------------------------------------------
xvars <- c(names(phytoExpr_dirs), names(empres_files))
yvar <- "log_tpm"
out <- data.frame()

label_DF <- data.frame()
plot_DF <- data.frame()

#--------------------------
# Between-gene differences
#--------------------------
# Averages among controls
controls <- pred[startsWith(pred$id, "C"), ]
controls <- aggregate(controls[, c(xvars, yvar)], list('gene'=controls$gene), mean)

# Validation
for (xvar in xvars) {
  
  print(xvar)
  
  # Linear regression
  fit <- paste(yvar, "~", xvar) %>%
    as.formula() %>%
    lm(data=controls)
  
  fit.out <- tidy(fit) %>%
    as.data.frame() %>%
    dplyr::filter(term == xvar) %>%
    cbind(variation="between")
  
  fit.out$partial_r2 <- partial_r2(fit)[xvar]
  
  out <- rbind(out, fit.out)
  
  
  # Graphics
  beta_label <- paste("beta ==", signif(fit.out$estimate, 2))
  p_label <- ifelse(fit.out$p.value == 0,
                    "P < 2.2e-16",
                    paste("P ==", format(signif(fit.out$p.value, 2))))
  
  label_DF <- rbind(label_DF,
                    data.frame(
                      variation="between",
                      model=xvar,
                      x=min(controls[, xvar]),
                      y=max(controls[, yvar]),
                      label=paste0(beta_label, "~(", p_label, ")")
                    ))
  
  plot_DF <- rbind(plot_DF,
                   data.frame(
                     variation="between",
                     model=xvar,
                     pred=controls[, xvar],
                     obs=controls[, yvar]
                   ))
  
}

#--------------------------
# Within-gene differences
#--------------------------
# Predictions from controls and mutants
mutants <- data.table(pred[startsWith(pred$id, "M"), ]) %>%
  merge(data.table(controls), by="gene", suffixes=c(".mutant", ".control")) %>%
  dplyr::mutate(delta.obs=log_tpm.mutant-log_tpm.control) %>%
  as.data.frame()

mutants <- mutants[mutants$case %in% unique_cases, ] %>%
  merge(DF_PCA, by='id') %>%
  merge(DF_PEER, by='id')

# Null model
BIC_seq <- sapply(1:max_PCs, function(k) {
  
  fit <- paste(c("delta.obs ~", paste0("PC", 1:k)), collapse=" + ") %>%
    as.formula() %>%
    lm(data=mutants)
  
  return(BIC(fit))
  
})

n_PCs <- (1:max_PCs)[which.min(BIC_seq)]

res.obs <- paste(c("delta.obs ~ ", paste0("PC", 1:n_PCs)), collapse=" + ") %>%
  as.formula() %>%
  lm(data=mutants) %>%
  residuals()

# Validation
for (xvar in xvars) {
  
  print(xvar)
  
  mutants$delta.pred <- mutants[, paste0(xvar, ".mutant")] - mutants[, paste0(xvar, ".control")]
  
  res.pred <- paste(c("delta.pred ~ ", paste0("X", 1:10)), collapse=" + ") %>%
    as.formula() %>%
    lm(data=mutants) %>%
    residuals()
  
  # Linear regression
  fit <- paste(c("delta.obs ~ delta.pred", paste0("PC", 1:n_PCs)), collapse=" + ") %>%
    as.formula() %>%
    lm(data=mutants)
  
  fit.out <- tidy(fit) %>%
    as.data.frame() %>%
    dplyr::filter(term == "delta.pred") %>%
    dplyr::mutate(term=.GlobalEnv$xvar) %>% 
    cbind(variation="within")
  
  fit.out$partial_r2 <- partial_r2(fit)["delta.pred"]
  
  out <- rbind(out, fit.out)
  
  # Graphics
  beta_label <- paste("beta ==", signif(fit.out$estimate, 2))
  p_label <- ifelse(fit.out$p.value == 0,
                    "P < 2.2e-16",
                    paste("P ==", format(signif(fit.out$p.value, 2))))
  
  label_DF <- rbind(label_DF,
                    data.frame(
                      variation="within",
                      model=xvar,
                      x=min(mutants$delta.pred),
                      y=max(mutants$delta.obs),
                      label=paste0(beta_label, "~(", p_label, ")")
                    ))
  
  plot_DF <- rbind(plot_DF,
                   data.frame(
                     variation="within",
                     model=xvar,
                     pred=mutants$delta.pred,
                     obs=mutants$delta.obs
                   ))
  
}

# Output
dplyr::rename(out, model=term) %>%
  dplyr::select(variation, model, estimate, std.error, statistic, p.value, partial_r2) %>%
  write.csv(output_file, row.names=FALSE)

saveRDS(plot_DF, plot_file)
saveRDS(label_DF, label_file)

#---------------------
# Graphics
#---------------------
# Histograms of between- and within-gene variation
hist_DF <- rbind(
  data.frame(variation="Between-gene variation\n(average among controls)",
             x=controls$log_tpm),
  data.frame(variation="Within-gene variation\n(difference between mutants and controls)",
             x=mutants$delta.obs)
)

hist.out <- ggplot(hist_DF, aes(x)) +
  geom_histogram(fill="#3399FF", color="black", binwidth=0.05, linewidth=0.2) +
  labs(x=NULL,
       y=NULL) +
  theme_bw(base_size=34) +
  facet_wrap(~ variation, ncol=1, scales="free")

ggsave(
  paste0(fig_dir, "Figure_3b.png"),
  plot=hist.out,
  device="png",
  scale=1,
  width=10,
  height=16,
  units="in",
  dpi=300,
  limitsize=TRUE,
  bg=NULL,
)

# Scatter plots
model_names <- c("phytoExpr_B"="phytoExpr B",
                 "phytoExpr_C"="phytoExpr C",
                 "EMPRES_1"="EMPRES 1",
                 "EMPRES_2"="EMPRES 2",
                 "EMPRES_3"="EMPRES 3",
                 "EMPRES_4"="EMPRES 4")

focus <- function(DF) {
  DF %>%
    dplyr::mutate(model=gsub("_", " ", model)) %>%
    dplyr::filter(model %in% c("phytoExpr C", "EMPRES 2")) %>%
    dplyr::mutate(model=factor(model, levels=c("phytoExpr C", "EMPRES 2")),
                  variation=paste0(capitalize(variation),"-gene variation"))
}

plot.out <- ggplot(focus(plot_DF), aes(x=pred, y=obs)) +
  geom_point(alpha=0.15) +
  geom_smooth(method="lm", formula=y ~ x) +
  xlab("Predicted") +
  ylab("Observed") +
  theme_bw(base_size=28) +
  geom_label(data=focus(label_DF),
             mapping=aes(x=x, y=y, label=label),
             hjust="left",
             size=7,
             parse=TRUE) +
  facet_wrap(model ~ variation, scales="free")

ggsave(
  paste0(fig_dir, "Figure_6.png"),
  plot=plot.out,
  device="png",
  scale=1,
  width=16,
  height=16,
  units="in",
  dpi=300,
  limitsize=TRUE,
  bg=NULL,
)

ggsave(
  paste0(fig_dir, "Figure_6-uncompressed.tiff"),
  plot=plot.out,
  device="tiff",
  scale=1,
  width=16,
  height=16,
  units="in",
  dpi=300,
  limitsize=TRUE,
  bg=NULL,
)

tiff(paste0(fig_dir, "Figure_6.tiff"), width=16, height=16, units="in", compression="lzw", res=300)
print(plot.out)
dev.off()
