#!/usr/bin/env Rscript
# Seurat Cell Cycle Scoring for RAW GEX Data
# Input: RAW count CSV files from multiomics_extractor_RAW.R
# Output: Same files with "phase" column added (G1, G2M, S)

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(tibble)
})

# Configuration
CONFIG <- list(
  # Input directory (where RAW GEX files are)
  input_dir = "/data/halimaakhter/TF_Rscript/scMultiom_data/",

  # Output directory (can be same as input)
  output_dir = "/data/halimaakhter/TF_Rscript/scMultiom_data/",

  # Train/test split ratio (0.0 = no split, 0.2 = 20% test)
  test_split_ratio = 0.2,

  # Random seed for reproducibility
  random_seed = 42,

  verbose = TRUE
)

# Utility function
log_message <- function(message, verbose = CONFIG$verbose) {
  if (verbose) {
    cat(paste0("[", Sys.time(), "] ", message, "\n"))
  }
}

perform_cell_cycle_scoring <- function(gex_file) {
 
  #Perform Seurat cell cycle scoring on RAW GEX data.

  #Args:
    #gex_file: Path to RAW GEX CSV file

  #Returns:
    #DataFrame with cell_id, genes, and phase column


  log_message(sprintf("Processing: %s", basename(gex_file)))

  # Read RAW counts
  log_message("  Reading RAW count data...")
  gex_df <- read.csv(gex_file, row.names = 1, check.names = FALSE)

  log_message(sprintf("  Loaded: %d cells x %d genes", nrow(gex_df), ncol(gex_df)))

  # Transpose to genes x cells format for Seurat
  log_message("  Transposing to genes x cells...")
  count_matrix <- t(as.matrix(gex_df))

  # Create Seurat object
  log_message("  Creating Seurat object...")
  seurat_obj <- CreateSeuratObject(
    counts = count_matrix,
    project = "CellCycleScoring",
    min.cells = 0,  # Keep all genes
    min.features = 0  # Keep all cells
  )

  log_message(sprintf("  Seurat object: %d genes x %d cells", nrow(seurat_obj), ncol(seurat_obj)))

  # Normalize data (required for cell cycle scoring)
  log_message("  Normalizing for cell cycle scoring...")
  seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)

  # Cell cycle scoring
  log_message("  Running CellCycleScoring...")

  # Load cell cycle genes (built-in Seurat markers)
  s.genes <- cc.genes$s.genes
  g2m.genes <- cc.genes$g2m.genes

  log_message(sprintf("  Using %d S-phase genes and %d G2M-phase genes",
                      length(s.genes), length(g2m.genes)))

  # Score cells
  seurat_obj <- CellCycleScoring(
    seurat_obj,
    s.features = s.genes,
    g2m.features = g2m.genes,
    set.ident = TRUE
  )

  # Extract phase assignments
  phase_assignments <- seurat_obj@meta.data$Phase

  log_message("  Cell cycle phase distribution:")
  phase_table <- table(phase_assignments)
  for (phase in names(phase_table)) {
    log_message(sprintf("    %s: %d cells (%.1f%%)",
                        phase, phase_table[phase],
                        phase_table[phase]/sum(phase_table)*100))
  }

  # Add phase column to original dataframe
  gex_df$phase <- phase_assignments

  # Reorder columns (phase at the end)
  gex_df <- gex_df %>%
    select(-phase, everything(), phase)

  # Add cell_id as first column
  gex_df <- rownames_to_column(gex_df, var = "cell_id")

  log_message(sprintf("  Final dataframe: %d cells x %d features (including phase)",
                      nrow(gex_df), ncol(gex_df)))

  return(gex_df)
}

split_train_test <- function(df, test_ratio = CONFIG$test_split_ratio, seed = CONFIG$random_seed) {
  
  #Split dataframe into train and test sets, stratified by phase.

  #Args:
    #df: DataFrame with phase column
    #test_ratio: Fraction of data for test set
    #seed: Random seed

  #Returns:
    #List with train_df and test_df
  

  if (test_ratio == 0.0) {
    log_message("  No train/test split requested (test_ratio = 0.0)")
    return(list(train = df, test = NULL))
  }

  log_message(sprintf("  Splitting data: %.0f%% train, %.0f%% test",
                      (1-test_ratio)*100, test_ratio*100))

  set.seed(seed)

  # Stratified split by phase
  train_indices <- c()
  test_indices <- c()

  for (phase in unique(df$phase)) {
    phase_indices <- which(df$phase == phase)
    n_test <- round(length(phase_indices) * test_ratio)

    test_idx <- sample(phase_indices, n_test)
    train_idx <- setdiff(phase_indices, test_idx)

    train_indices <- c(train_indices, train_idx)
    test_indices <- c(test_indices, test_idx)
  }

  train_df <- df[train_indices, ]
  test_df <- df[test_indices, ]

  log_message(sprintf("  Train set: %d cells", nrow(train_df)))
  log_message("    Phase distribution:")
  for (phase in names(table(train_df$phase))) {
    count <- sum(train_df$phase == phase)
    log_message(sprintf("      %s: %d", phase, count))
  }

  log_message(sprintf("  Test set: %d cells", nrow(test_df)))
  log_message("    Phase distribution:")
  for (phase in names(table(test_df$phase))) {
    count <- sum(test_df$phase == phase)
    log_message(sprintf("      %s: %d", phase, count))
  }

  return(list(train = train_df, test = test_df))
}

save_scored_data <- function(df_list, base_filename, output_dir = CONFIG$output_dir) {


  # Create output directory if needed
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Extract base name (remove _GEX_RAW.csv)
  base_name <- gsub("_GEX_RAW\\.csv$", "", base_filename)

  # Save full dataset with phase labels
  full_output <- file.path(output_dir, paste0(base_name, "_GEX_RAW_cellcycle.csv"))

  if (!is.null(df_list$test)) {
    # Combine train and test for full dataset
    full_df <- rbind(df_list$train, df_list$test)
  } else {
    full_df <- df_list$train
  }

  log_message(sprintf("  Saving full dataset: %s", full_output))
  write.csv(full_df, full_output, row.names = FALSE)

  # Save train set
  train_output <- file.path(output_dir, paste0(base_name, "_GEX_RAW_cellcycle_training.csv"))
  log_message(sprintf("  Saving training set: %s", train_output))
  write.csv(df_list$train, train_output, row.names = FALSE)

  # Save test set if exists
  if (!is.null(df_list$test)) {
    test_output <- file.path(output_dir, paste0(base_name, "_GEX_RAW_cellcycle_test.csv"))
    log_message(sprintf("  Saving test set: %s", test_output))
    write.csv(df_list$test, test_output, row.names = FALSE)
  }

  return(full_output)
}

# Main processing function
process_all_gex_files <- function() {
  log_message(strrep("=", 80))
  log_message("Seurat Cell Cycle Scoring Pipeline")
  log_message(strrep("=", 80))

  log_message(sprintf("Input directory: %s", CONFIG$input_dir))
  log_message(sprintf("Output directory: %s", CONFIG$output_dir))
  log_message(sprintf("Test split ratio: %.1f%%", CONFIG$test_split_ratio * 100))

  # Find all GEX RAW files
  gex_files <- list.files(
    CONFIG$input_dir,
    pattern = "_GEX_RAW\\.csv$",
    full.names = TRUE
  )

  log_message(sprintf("\nFound %d GEX RAW files:", length(gex_files)))
  for (f in gex_files) {
    log_message(sprintf("  - %s", basename(f)))
  }

  if (length(gex_files) == 0) {
    stop("No GEX_RAW.csv files found in input directory!")
  }

  # Process each file
  results <- list()

  for (i in seq_along(gex_files)) {
    gex_file <- gex_files[i]

    log_message(sprintf("\n=== Processing file %d/%d ===", i, length(gex_files)))

    tryCatch({
      # Cell cycle scoring
      scored_df <- perform_cell_cycle_scoring(gex_file)

      # Train/test split
      split_data <- split_train_test(scored_df)

      # Save results
      output_file <- save_scored_data(split_data, basename(gex_file))

      results[[basename(gex_file)]] <- list(
        n_cells = nrow(scored_df),
        n_genes = ncol(scored_df) - 2,  # Exclude cell_id and phase
        phase_dist = table(scored_df$phase),
        output_file = output_file
      )

      log_message("  SUCCESS!")

    }, error = function(e) {
      log_message(sprintf("  ERROR: %s", e$message))
      traceback()
    })
  }

  # Print summary
  log_message(strrep("\n", 2))
  log_message(strrep("=", 80))
  log_message("PROCESSING COMPLETE - SUMMARY")
  log_message(strrep("=", 80))

  for (filename in names(results)) {
    result <- results[[filename]]
    log_message(sprintf("\n%s:", filename))
    log_message(sprintf("  Cells: %d", result$n_cells))
    log_message(sprintf("  Genes: %d", result$n_genes))
    log_message("  Phase distribution:")
    for (phase in names(result$phase_dist)) {
      log_message(sprintf("    %s: %d (%.1f%%)",
                          phase, result$phase_dist[phase],
                          result$phase_dist[phase]/sum(result$phase_dist)*100))
    }
    log_message(sprintf("  Output: %s", basename(result$output_file)))
  }

  log_message(strrep("\n", 2))
  log_message("Files ready for DANN training!")
  log_message(strrep("=", 80))

  return(results)
}

# MAIN EXECUTION
log_message("\nStarting cell cycle scoring pipeline...")
results <- process_all_gex_files()

# Save summary to file
summary_file <- file.path(CONFIG$output_dir, "cellcycle_scoring_summary.txt")
sink(summary_file)
cat("=== CELL CYCLE SCORING SUMMARY ===\n\n")
for (filename in names(results)) {
  result <- results[[filename]]
  cat(sprintf("%s:\n", filename))
  cat(sprintf("  Cells: %d\n", result$n_cells))
  cat(sprintf("  Genes: %d\n", result$n_genes))
  cat("  Phase distribution:\n")
  for (phase in names(result$phase_dist)) {
    cat(sprintf("    %s: %d (%.1f%%)\n",
                phase, result$phase_dist[phase],
                result$phase_dist[phase]/sum(result$phase_dist)*100))
  }
  cat("\n")
}
sink()

log_message(sprintf("\nSummary saved to: %s", summary_file))
