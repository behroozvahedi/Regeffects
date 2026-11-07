# Validation of prediction models
# Objective : Compare observed and predicted values in RNA abundance
# Created by: Guillaume Ramstein (ramstein@qgg.au.dk)
# Created on: 7/2/2025

#--------------------------------------------------------
# Script parameters
#--------------------------------------------------------
# Working directory
setwd("~/sieve/WP2/OUTPUT/sieve/validation/")

# Input files
controls_file <- "between_gene_variation.rds"
mutants_file <- "within_gene_variation.rds"
count_file <- "~/sieve/WP2/OUTPUT/sieve/mutation_counts.tsv"

# Validation
# xvars <- c(
#   "phytoExpr_B",
#   "phytoExpr_C",
#   "phytoExpr_B_July2026",
#   "phytoExpr_C_July2026",
#   "EMPRES_0",
#   "EMPRES_0_July2026",
#   "EMPRES_1",
#   "EMPRES_2",
#   "EMPRES_3",
#   "EMPRES_4"
# )

xvars <- c(
  "phytoExpr_B",
  "phytoExpr_C",
  "EMPRES_0",
  "EMPRES_1",
  "EMPRES_2",
  "EMPRES_3",
  "EMPRES_4"
)

max_PCs <- 20

OLS <- TRUE


# Output
method_tag <- ifelse(OLS, "_OLS", "_M")

input_file <- paste0("validation_input", method_tag, ".rds")
output_file <- paste0("validation_results", method_tag, ".rds")
results_file <- paste0("regression_results", method_tag, ".csv")

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

test_mutants <- function(DF, xvar, covariates=NULL, weights=NULL, OLS=TRUE) {
  
  f <- paste0("delta.obs ~ ", xvar)
  
  if (!is.null(covariates)) f <- paste(c(f, covariates), collapse=" + ")
  
  # Linear regression
  if (OLS) {
    
    fit <- lm(as.formula(f), weights=weights, data=DF)
    
    fit.out <- tidy(fit) %>%
      as.data.frame() %>%
      dplyr::filter(term == .env$xvar) %>%
      cbind(variation="within")
    
    fit.out$partial_r2 <- partial_r2(fit)[xvar]
    
  } else {
    
    fit <- MASS::rlm(as.formula(f), weights=weights, data=DF, maxit=100)
    
    if (! fit$converged) warning("rlm did not converge for ", xvar, call.=FALSE)
    
    fit.sum <- summary(fit)
    coefs <- fit.sum$coefficients
    rdf <- fit.sum$df[2]
    t_value <- coefs[xvar, "t value"]
    
    fit.out <- data.frame(term=xvar,
                          estimate=coefs[xvar, "Value"],
                          std.error=coefs[xvar, "Std. Error"],
                          statistic=t_value,
                          p.value=2 * pt(-abs(t_value), df=rdf),
                          variation="within",
                          stringsAsFactors=FALSE)
    
    fit.out$partial_r2 <- t_value^2 / (t_value^2 + rdf)
    
  }
  
  return(fit.out)
  
}

HOM_MUTATIONS <- c("A>T", "A>C", "A>G",
                   "T>A", "T>C", "T>G",
                   "C>A", "C>T", "C>G",
                   "G>A", "G>T", "G>C")

#--------------------------------------------------------
# Data
#--------------------------------------------------------
# Between-gene variation
controls <- readRDS(controls_file)

# Within-gene variation
mutants <- readRDS(mutants_file)

# Mutation counts
counts <- fread(count_file) %>%
  dplyr::mutate(case=paste(transcript, id, sep="_")) %>%
  as.data.frame()

MUTATIONS <- colnames(counts)[grep(">", colnames(counts))]
HET_MUTATIONS <- setdiff(MUTATIONS, HOM_MUTATIONS)

counts$n_het <- rowSums(counts[, HET_MUTATIONS])
counts$n <- rowSums(counts[, HOM_MUTATIONS]) + 0.5 * counts$n_het

counts$n_GC <- rowSums(counts[, c("G>A", "C>T")])

# Subsetting: only cases with mutations
mutants <- mutants[mutants$case %in% counts$case[counts$n > 0], ]

# Load
mutants$load <- tapply(counts$n, counts$id, sum)[mutants$id]

gc()

# Making delta values
for (xvar in xvars) {
  
  mutants[, xvar] <- mutants[, paste0(xvar, ".mutant")] - mutants[, paste0(xvar, ".control")]
  
}

# # Subsetting: only homozygous mutations
hom_mutants  <- mutants[mutants$case %in% counts$case[counts$n_het == 0], ]

GC_mutants <- hom_mutants[hom_mutants$case %in% counts$case[counts$n_GC > 0 & counts$n_GC == counts$n], ]

#--------------------------------------------------------
# Model evaluation
#--------------------------------------------------------
res <- list()

#--------------------------
# Within-gene differences
#--------------------------
BIC_seq <- sapply(0:max_PCs, function(n_PCs) {
  
  if (n_PCs == 0) PCs <- NULL else PCs <- paste0("PC", 1:n_PCs)
  
  fit <- paste(c("delta.obs ~ 1", PCs), collapse=" + ") %>%
    as.formula() %>%
    lm(data=GC_mutants)
  
  return(BIC(fit))
  
})

optimal_PCs <- (0:max_PCs)[which.min(BIC_seq)]
cat("Optimal number of PCs:", optimal_PCs, "\n")

res[["optimal_PCs"]] <- optimal_PCs

res[["base"]] <- foreach(xvar=xvars, .combine=rbind) %do% {
  
  print(xvar)
  
  if (optimal_PCs == 0) PCs <- NULL else PCs <- paste0("PC", 1:optimal_PCs)
  
  test_mutants(GC_mutants, xvar, PCs, OLS=OLS)
  
}

#--------------------------
# Accounting for trans effects
#--------------------------
GC_mutants$L_class <- ntile(GC_mutants$load, 4)

res[["trans"]] <- foreach(xvar=xvars, .combine=rbind) %do% {
  
  print(xvar)
  
  foreach(L_class=sort(unique(GC_mutants$L_class)), .combine=rbind) %do% {
    
    if (optimal_PCs == 0) PCs <- NULL else PCs <- paste0("PC", 1:optimal_PCs)
    
    out <- test_mutants(GC_mutants[GC_mutants$L_class == L_class, ], xvar, PCs, OLS=OLS)
    
    cbind(L_class=L_class, out)
    
  }
  
}

#--------------------------
# Different numbers of PCs
#--------------------------
res[["PCs"]] <- foreach(xvar=xvars, .combine=rbind) %do% {
  
  print(xvar)
  
  foreach(n_PCs=0:max_PCs, .combine=rbind) %do% {
    
    if (n_PCs == 0) PCs <- NULL else PCs <- paste0("PC", 1:n_PCs)
    
    out <- test_mutants(GC_mutants, xvar, PCs, OLS=OLS)
    
    cbind(out, n_PCs=n_PCs)
    
  }
  
}

#--------------------------
# Any homozygous mutation
#--------------------------
res[["all"]] <- foreach(xvar=xvars, .combine=rbind) %do% {
  
  print(xvar)
  
  if (optimal_PCs == 0) PCs <- NULL else PCs <- paste0("PC", 1:optimal_PCs)
  
  test_mutants(hom_mutants, xvar, PCs, OLS=OLS)
  
}

#--------------------------
# Between-gene differences
#--------------------------
res[["between"]] <- foreach(xvar=xvars, .combine=rbind) %do% {
  
  print(xvar)
  
  # Linear regression
  fit <- paste("log_tpm ~", xvar) %>%
    as.formula() %>%
    lm(data=controls)
  
  fit.out <- tidy(fit) %>%
    as.data.frame() %>%
    dplyr::filter(term == xvar) %>%
    cbind(variation="between")
  
  fit.out$partial_r2 <- partial_r2(fit)[xvar]
  
  fit.out
  
}


#--------------------------------------------------------
# Output
#--------------------------------------------------------
# Data
saveRDS(list("controls"=controls,
             "mutants"=mutants,
             "hom_mutants"=hom_mutants,
             "GC_mutants"=GC_mutants),
        input_file)

# Results
saveRDS(res, output_file)

# Baseline regression results
rbind(res[["between"]], res[["base"]]) %>%
dplyr::rename(model=term) %>%
  dplyr::select(variation,
                model,
                estimate,
                std.error,
                statistic,
                p.value,
                partial_r2) %>%
  write.csv(results_file, row.names=FALSE)

