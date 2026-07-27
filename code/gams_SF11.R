# ============================================================
# Developmental GAMs across MIND feature sets
#   1. Test significance of non-linear age term for every region
#   2. Plot example fit + first derivative
# Outputs -> output/gam/
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(mgcv)       # GAM
  library(gratia)     # derivatives
  library(patchwork)  # subplots
})

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
data_dir <- "/Users/EdHutchings_1/proj1/marm_MIND/output/subj_dfs"
out_dir  <- "output/gam"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- list(
  list(name = "mean_t12", file = "mean_t12_per_subj_filt.csv", roi = "AuA1_L"),
  list(name = "degree",   file = "degree_per_subj_filt.csv",   roi = "AuA1_L"),
  list(name = "edge",     file = "edge_per_subj_filt.csv",     roi = "AuA1_L_AI_R")
)

age_filt      <- FALSE   # hard cut on developmental age
age_threshold <- 4
density_filt  <- TRUE    # drop ages with < 5% data density
first_region_col <- 6    # regions are columns first_region_col:ncol
k_basis       <- 10
n_grid        <- 200

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
prepare_data <- function(path) {
  d <- read.csv(path)
  d$Sex <- as.factor(d$Sex)

  if (age_filt) d <- d[d$Age < age_threshold, ]

  if (density_filt) {
    # Automated smoothness selection is poor when data are sparse
    # (https://cran.r-project.org/web/packages/mgcv/mgcv.pdf)
    dens  <- density(d$Age)
    valid <- dens$x[dens$y > 0.05 * max(dens$y)]
    d <- d[d$Age < max(valid), ]
  }
  d
}

#' Index of the first age at which the derivative is credibly shallower
#' than its value at the start of the curve.
age_at_slowing <- function(deriv) {
  max_slope <- deriv$.derivative[1]
  if (max_slope > 0) {
    sig_less_steep <- (deriv$.upper_ci - max_slope) < 0
  } else {
    sig_less_steep <- (deriv$.lower_ci - max_slope) > 0
  }
  deriv$Age[which(sig_less_steep)[1]]
}

#' Start/end indices of the first contiguous run of credible change.
change_window <- function(deriv) {
  sig <- with(deriv, (.lower_ci > 0 & .upper_ci > 0) | (.lower_ci < 0 & .upper_ci < 0))
  i_start <- which(sig)[1]
  if (is.na(i_start)) return(list(start = NA_integer_, end = NA_integer_))
  i_end <- i_start + which(!sig[i_start:length(sig)])[1] - 2
  if (is.na(i_end)) i_end <- length(sig)   # still changing at the last age
  list(start = i_start, end = i_end)
}

# ------------------------------------------------------------
# 1. Significance of the non-linear age term, per region
# ------------------------------------------------------------
fit_all_regions <- function(d) {
  regions <- colnames(d)[first_region_col:ncol(d)]
  results <- vector("list", length(regions))

  for (i in seq_along(regions)) {
    region <- regions[i]

    # Shrinkage-free linear term + pure-wiggliness smooth -> tests non-linearity
    f_test <- as.formula(sprintf("%s ~ Age + s(Age, k=%d, m=c(2,0)) + Sex", region, k_basis))
    m_test <- gam(f_test, data = d, method = "REML")

    p_lin_age <- summary(m_test)$p.table["Age", "Pr(>|t|)"]
    p_smo_age <- summary(m_test)$s.table[1, "p-value"]

    edf <- slowing_age <- no_change_age <- tp_age <- NA_real_

    if (!is.na(p_smo_age) && p_smo_age < 0.05) {
      f_full <- as.formula(sprintf("%s ~ s(Age, k=%d) + Sex", region, k_basis))
      m_full <- gam(f_full, data = d, method = "REML")
      edf <- summary(m_full)$s.table[1, "edf"]

      # Simultaneous intervals for the credible-change criteria
      # https://fromthebottomoftheheap.net/2014/05/15/identifying-periods-of-change-with-gams/
      deriv_sim <- derivatives(m_full, term = 1, n = n_grid, interval = "simultaneous")

      slowing_age <- age_at_slowing(deriv_sim)

      win <- change_window(deriv_sim)
      no_change_age <- if (is.na(win$end)) NA_real_ else deriv_sim$Age[win$end]

      # Turning point: first sign flip of the pointwise derivative
      deriv_pt   <- derivatives(m_full, term = 1, n = n_grid)
      sign_change <- diff(sign(deriv_pt$.derivative))
      tp_age <- deriv_pt$Age[which(sign_change != 0)][1]
    }

    results[[i]] <- data.frame(
      region         = region,
      p_lin_age      = p_lin_age,
      p_smo_age      = p_smo_age,
      edf            = edf,
      slowing_age    = slowing_age,
      no_change_age  = no_change_age,
      tp             = tp_age,
      stringsAsFactors = FALSE
    )
  }

  results <- bind_rows(results)
  results$p_lin_age_fdr <- p.adjust(results$p_lin_age, method = "fdr")
  results$p_smo_age_fdr <- p.adjust(results$p_smo_age, method = "fdr")
  results
}

# ------------------------------------------------------------
# 2. Example GAM + derivative, with credible-change window marked
# ------------------------------------------------------------
base_theme <- theme_bw() +
  theme(
    plot.title       = element_text(size = 12),
    legend.position  = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

#' Split a data frame into pre / during / post change segments.
split_segments <- function(df, i_min, i_max) {
  list(
    pre    = df[1:i_min, ],
    during = df[i_min:i_max, ],
    post   = df[i_max:nrow(df), ]
  )
}

plot_example <- function(d, region, tag,
                         highlight_fit = FALSE, highlight_deriv = TRUE) {

  f <- as.formula(sprintf("%s ~ s(Age, k=%d) + Sex", region, k_basis))
  m <- gam(f, data = d, method = "REML")

  pred_data <- data.frame(
    Age = seq(min(d$Age), max(d$Age), length.out = n_grid),
    Sex = factor(levels(d$Sex)[1], levels = levels(d$Sex))
  )
  pred <- predict(m, newdata = pred_data, se.fit = TRUE, type = "response")
  pred_data$fit   <- pred$fit
  pred_data$upper <- pred$fit + 1.96 * pred$se.fit
  pred_data$lower <- pred$fit - 1.96 * pred$se.fit

  deriv <- derivatives(m, term = 1, n = n_grid, interval = "simultaneous")
  win   <- change_window(deriv)
  has_window <- !is.na(win$start) && !is.na(win$end) && win$end > win$start

  if (has_window) {
    # Map derivative-grid indices onto the prediction grid
    i_min_p <- which.min(abs(pred_data$Age - deriv$Age[win$start]))
    i_max_p <- which.min(abs(pred_data$Age - deriv$Age[win$end]))
    pred_seg  <- split_segments(pred_data, i_min_p, i_max_p)
    deriv_seg <- split_segments(deriv, win$start, win$end)
  }

  # --- fitted curve ---
  fit_layers <- if (highlight_fit && has_window) {
    list(
      geom_ribbon(data = pred_seg$pre,    aes(Age, ymin = lower, ymax = upper), fill = "black", alpha = 0.2),
      geom_ribbon(data = pred_seg$during, aes(Age, ymin = lower, ymax = upper), fill = "red",   alpha = 0.2),
      geom_ribbon(data = pred_seg$post,   aes(Age, ymin = lower, ymax = upper), fill = "black", alpha = 0.2),
      geom_line(data = pred_seg$pre,      aes(Age, fit), colour = "black", linewidth = 1),
      geom_line(data = pred_seg$during,   aes(Age, fit), colour = "red",   linewidth = 1),
      geom_line(data = pred_seg$post,     aes(Age, fit), colour = "black", linewidth = 1)
    )
  } else {
    list(
      geom_ribbon(data = pred_data, aes(Age, ymin = lower, ymax = upper), fill = "black", alpha = 0.2),
      geom_line(data = pred_data,   aes(Age, fit), colour = "black", linewidth = 1)
    )
  }

  p_fit <- ggplot() + fit_layers +
    scale_x_continuous(breaks = 1:10) +
    labs(title = region, x = "Age", y = "Predicted values") +
    base_theme

  # --- first derivative ---
  deriv_layers <- if (highlight_deriv && has_window) {
    list(
      geom_ribbon(data = deriv_seg$pre,    aes(Age, ymin = .lower_ci, ymax = .upper_ci), fill = "black", alpha = 0.2),
      geom_ribbon(data = deriv_seg$during, aes(Age, ymin = .lower_ci, ymax = .upper_ci), fill = "red",   alpha = 0.2),
      geom_ribbon(data = deriv_seg$post,   aes(Age, ymin = .lower_ci, ymax = .upper_ci), fill = "black", alpha = 0.2),
      geom_line(data = deriv_seg$pre,      aes(Age, .derivative), colour = "black", linewidth = 1),
      geom_line(data = deriv_seg$during,   aes(Age, .derivative), colour = "red",   linewidth = 1),
      geom_line(data = deriv_seg$post,     aes(Age, .derivative), colour = "black", linewidth = 1)
    )
  } else {
    list(
      geom_ribbon(data = deriv, aes(Age, ymin = .lower_ci, ymax = .upper_ci), fill = "black", alpha = 0.2),
      geom_line(data = deriv,   aes(Age, .derivative), colour = "black", linewidth = 1)
    )
  }

  p_deriv <- ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    deriv_layers +
    scale_x_continuous(breaks = 1:10) +
    labs(title = region, x = "Age", y = "Derivative") +
    base_theme

  plot_grid <- wrap_plots(list(p_fit, p_deriv), ncol = 1)
  outfile <- file.path(out_dir, sprintf("%s_%s_plots.pdf", tag, region))
  ggsave(outfile, plot = plot_grid, width = 3, height = 4)
  outfile
}

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
for (ds in datasets) {
  message("=== ", ds$name, " ===")

  d <- prepare_data(file.path(data_dir, ds$file))

  print(d %>% summarise(N_total  = n(),
                        N_female = sum(Sex == 0),
                        Max_age  = max(Age)))

  results <- fit_all_regions(d)
  res_file <- file.path(out_dir, sprintf("%s_gam_results.csv", ds$name))
  write.csv(results, res_file, row.names = FALSE)
  message("  wrote ", res_file)

  fig_file <- plot_example(d, ds$roi, ds$name)
  message("  wrote ", fig_file)
}

message("Done.")
