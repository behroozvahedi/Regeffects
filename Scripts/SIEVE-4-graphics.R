# Validation of prediction models
# Objective : Compare observed and predicted values in RNA abundance
# Created by: Guillaume Ramstein (ramstein@qgg.au.dk)
# Created on: 7/2/2025
# [edited] Lines marked [new] or [edited] differ from the previous version.

#--------------------------------------------------------
# Script parameters
#--------------------------------------------------------
# Working directory
setwd("~/sieve/WP2/OUTPUT/sieve/validation/")

# Estimator used in SIEVE-3-validation.R
OLS <- TRUE
method_tag <- ifelse(OLS, "_OLS", "_M")

# Input files
input_file <- paste0("validation_input", method_tag, ".rds")
results_file <- paste0("validation_results", method_tag, ".rds")

# Selected models
xvars <- c(
  "phytoExpr_B",
  "phytoExpr_C",
  "EMPRES_0",
  "EMPRES_1",
  "EMPRES_2",
  "EMPRES_3",
  "EMPRES_4"
)

n_models <- length(xvars)
model_labels <- setNames(gsub("_", "", xvars), xvars)
model_colors <- c(phytoExpr_B="#999999",
                  phytoExpr_C="#000000",
                  EMPRES_0="#E69F00",
                  EMPRES_1="#009E73",
                  EMPRES_2="#3399FF",
                  EMPRES_3="#CC79A7",
                  EMPRES_4="#D55E00")[xvars]

F2_models <- c(
  "phytoExpr_B",
  "EMPRES_0",
  "EMPRES_2"
)

# Output
fig_dir <- "figures/"
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
# Observed and predicted values
DF_list <- readRDS(input_file)

controls <- DF_list[["controls"]]
GC_mutants <- DF_list[["GC_mutants"]]

# Regression results
res <- readRDS(results_file)

res_labels <- rbind(res[["between"]], res[["base"]])

beta_ylab <- expression("Prediction accuracy (" * beta * ")")

#--------------------------------------------------------
# F1: Histograms of between- and within-gene variation
#--------------------------------------------------------
# Input
DF1 <- rbind(
  data.frame(variation="Between-gene variation\n(average among controls)",
             x=controls$log_tpm),
  data.frame(variation="Within-gene variation\n(difference between mutants and controls)",
             x=GC_mutants$delta.obs)
)

# Output
F1 <- ggplot(DF1, aes(x)) +
  geom_histogram(fill="#3399FF", color="black", binwidth=0.05, linewidth=0.2) +
  labs(x=NULL, y=NULL) +
  theme_bw(base_size=34) +
  facet_wrap(~ variation, ncol=1, scales="free")

tiff(paste0(fig_dir, "SIEVE-Figure_1.tiff"),
     width=10,
     height=16,
     units="in",
     compression="lzw",
     res=300)
print(F1)
dev.off()

#--------------------------------------------------------
# F2: Scatter plots of observed vs. predicted values
#--------------------------------------------------------
# Input
DF2 <- data.frame()
label_DF <- data.frame()

for (xvar in F2_models) {

  for (var_type in c("between", "within")) {

    # Input
    if (var_type == "between") {

      yvar <- "log_tpm"

      DF <- controls

    } else {

      yvar <- "delta.obs"

      DF <- GC_mutants

    }

    # Labels
    out <- res_labels %>%
      dplyr::filter(variation == .env$var_type,
                    term == .env$xvar)

    beta_label <- paste("beta ==", signif(out$estimate, 2))
    p_label <- ifelse(out$p.value == 0,
                      "P < 2.2e-16",
                      paste("P ==", format(signif(out$p.value, 2))))

    label_DF <- rbind(label_DF,
                      data.frame(
                        variation=var_type,
                        model=xvar,
                        x=min(DF[, xvar], na.rm=TRUE),
                        y=max(DF[, yvar], na.rm=TRUE),
                        label=paste0(beta_label, "~(", p_label, ")")
                      ))

    # Data points
    out <- data.frame(obs=DF[, yvar],
                      pred=DF[, xvar],
                      model=xvar,
                      variation=var_type)

    DF2 <- rbind(DF2, out)

  }

}

# Formatting
DF2$variation <- factor(DF2$variation, levels=c("between", "within"))
DF2$model <- factor(DF2$model, levels=F2_models)

label_DF$variation <- factor(label_DF$variation, levels=c("between", "within"))
label_DF$model <- factor(label_DF$model, levels=F2_models)

num_cols <- length(unique(DF2$model))

variation_labels <- capitalize(levels(DF2$variation)) %>%
  paste0("-gene variation") %>%
  setNames(levels(DF2$variation))

# Output
F2 <- ggplot(DF2, aes(x=pred, y=obs)) +
  geom_point(alpha=0.15) +
  geom_smooth(method="lm", formula=y ~ x) +
  xlab("Predicted") +
  ylab("Observed") +
  theme_bw(base_size=28) +
  geom_label(data=label_DF,
             mapping=aes(x=x, y=y, label=label),
             hjust="left",
             size=7,
             parse=TRUE) +
  facet_wrap(~ variation + model,
             labeller=labeller(
               variation=variation_labels,
               model=model_labels
             ),
             scales="free",
             ncol=num_cols,
             drop=FALSE)

tiff(paste0(fig_dir, "SIEVE-Figure_2.tiff"),
     width=8*num_cols,
     height=16,
     units="in",
     compression="lzw",
     res=300)
print(F2)
dev.off()

#--------------------------------------------------------
# F3: Sensitivity analysis
#--------------------------------------------------------
q97.5 <- qnorm(0.975)

#--------------------------
# F3a: Any homozygous mutation vs. G:C>A:T transitions only
#--------------------------
subset_labels <- c("all"="Any homozygous mutation",
                   "base"="G:C>A:T transitions only")

DF3a <- rbind(
  cbind(res[["all"]], subset="all"),
  cbind(res[["base"]], subset="base")
) %>%
  dplyr::filter(term %in% .env$xvars) %>%
  dplyr::mutate(model=factor(term, levels=.env$xvars),
                subset=factor(subset, levels=names(.env$subset_labels)))

pd_bar <- position_dodge(width=0.8)

F3a <- ggplot(DF3a, aes(x=model, y=estimate, fill=subset)) +
  geom_hline(yintercept=0, linewidth=0.5) +
  geom_col(position=pd_bar, width=0.7, color="black", linewidth=0.2) +
  geom_errorbar(aes(ymin=estimate - q97.5 * std.error, ymax=estimate + q97.5 * std.error),
                position=pd_bar, width=0.25, linewidth=0.8) +
  scale_x_discrete(labels=function(x) model_labels[x]) +
  scale_fill_manual(values=c("all"="grey70", "base"="#3399FF"),
                    breaks=names(subset_labels), labels=unname(subset_labels)) +
  labs(x=NULL, y=beta_ylab, fill=NULL) +
  theme_bw(base_size=28) +
  theme(legend.position="top",
        axis.text.x=element_text(angle=45, hjust=1))

tiff(paste0(fig_dir, "SIEVE-Figure_3a.tiff"),
     width=12,
     height=10,
     units="in",
     compression="lzw",
     res=300)
print(F3a)
dev.off()


#--------------------------
# F3b: Load classes
#--------------------------
DF3b <- res[["trans"]] %>%
  dplyr::filter(term %in% .env$xvars) %>%
  dplyr::mutate(model=factor(term, levels=.env$xvars))

load_range <- tapply(GC_mutants$load, GC_mutants$L_class, range)
class_labels <- sapply(names(load_range), function(k) {
  r <- round(load_range[[k]], 1)
  paste0(k, "\n(", r[1], "-", r[2], ")")
})

pd_class <- position_dodge(width=0.5)

F3b <- ggplot(DF3b, aes(x=L_class, y=estimate, color=model, group=model)) +
  geom_hline(yintercept=0, linewidth=0.5) +
  geom_line(position=pd_class, linewidth=1) +
  geom_errorbar(aes(ymin=estimate - q97.5 * std.error, ymax=estimate + q97.5 * std.error),
                position=pd_class, width=0.5 / n_models, linewidth=0.8) +
  geom_point(position=pd_class, size=4) +
  scale_x_continuous(breaks=as.numeric(names(class_labels)), labels=class_labels) +
  scale_color_manual(values=model_colors, breaks=xvars, labels=unname(model_labels)) +
  guides(color=guide_legend(nrow=2)) +
  labs(x="Mutation load class (range of load)", y=beta_ylab, color=NULL) +
  theme_bw(base_size=28) +
  theme(legend.position="bottom")

tiff(paste0(fig_dir, "SIEVE-Figure_3b.tiff"),
     width=12,
     height=10,
     units="in",
     compression="lzw",
     res=300)
print(F3b)
dev.off()

#--------------------------
# F3c: Number of PCs
#--------------------------
DF3c <- res[["PCs"]] %>%
  dplyr::filter(term %in% .env$xvars) %>%
  dplyr::mutate(model=factor(term, levels=.env$xvars))

pd_pcs <- position_dodge(width=0.8)

F3c <- ggplot(DF3c, aes(x=n_PCs, y=estimate, color=model, group=model)) +
  geom_hline(yintercept=0, linewidth=0.5) +
  geom_line(position=pd_pcs, linewidth=1) +
  geom_errorbar(aes(ymin=estimate - q97.5 * std.error, ymax=estimate + q97.5 * std.error),
                position=pd_pcs, width=0.8 / n_models, linewidth=0.8) +
  geom_point(position=pd_pcs, size=3) +
  scale_x_continuous(breaks=seq(0, max(DF3c$n_PCs), by=2)) +
  scale_color_manual(values=model_colors, breaks=xvars, labels=unname(model_labels)) +
  guides(color=guide_legend(nrow=2)) +
  labs(x="Number of PCs", y=beta_ylab, color=NULL) +
  theme_bw(base_size=28) +
  theme(legend.position="bottom")

if (!is.null(res[["optimal_PCs"]])) {
  F3c <- F3c + geom_vline(xintercept=res[["optimal_PCs"]], linetype="dashed", linewidth=0.8)
}

tiff(paste0(fig_dir, "SIEVE-Figure_3c.tiff"),
     width=16,
     height=10,
     units="in",
     compression="lzw",
     res=300)
print(F3c)
dev.off()

#--------------------------
# F3: Combined figure
#--------------------------
F3_layout <- list(a=list(row=1, col=1),
                  b=list(row=1, col=2),
                  c=list(row=2, col=1:2))

# Panel letters
tag_theme <- theme(plot.tag=element_text(size=40, face="bold"))

tiff(paste0(fig_dir, "SIEVE-Figure_3.tiff"),
     width=24,
     height=20,
     units="in",
     compression="lzw",
     res=300)

F3_grobs <- list(
  a=ggplotGrob(F3a + labs(tag="a") + tag_theme),
  b=ggplotGrob(F3b + labs(tag="b") + tag_theme + theme(legend.position="none")),
  c=ggplotGrob(F3c + labs(tag="c") + tag_theme)
)

# Aligning the left edges of the plotting areas
left_cols <- function(g) {
  panel_col <- g$layout$l[g$layout$name == "panel"]
  if (length(panel_col) != 1) return(NULL)
  seq_len(panel_col - 1)
}

cols <- left_cols(F3_grobs$a)
if (!is.null(cols) && identical(cols, left_cols(F3_grobs$c))) {
  w <- grid::unit.pmax(F3_grobs$a$widths[cols], F3_grobs$c$widths[cols])
  F3_grobs$a$widths[cols] <- w
  F3_grobs$c$widths[cols] <- w
}

# Drawing
grid::grid.newpage()
grid::pushViewport(grid::viewport(layout=grid::grid.layout(nrow=2, ncol=2)))

for (p in names(F3_layout)) {
  grid::pushViewport(grid::viewport(layout.pos.row=F3_layout[[p]]$row,
                                    layout.pos.col=F3_layout[[p]]$col))
  grid::grid.draw(F3_grobs[[p]])
  grid::upViewport()
}

dev.off()
