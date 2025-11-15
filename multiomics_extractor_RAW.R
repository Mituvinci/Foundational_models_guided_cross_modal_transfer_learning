#!/usr/bin/env Rscript
# Modified scMultiomics Data Extractor - Outputs RAW COUNTS for consistent Python normalization
# CRITICAL CHANGE: normalize_rna and normalize_atac set to FALSE
# Python will do log1p(CPM) for BOTH modalities consistently

suppressPackageStartupMessages({
  library(Seurat)
  library(Signac)
  library(hdf5r)
  library(Matrix)
  library(dplyr)
  library(tibble)
})

# Configuration parameters
CONFIG <- list(
  # Filtering thresholds
  min_cells_per_gene = 1,
  min_genes_per_cell = 50,
  max_genes_per_cell = 10000,
  max_mt_percent = 50,
  min_peak_cells = 1,
  min_cells_per_peak = 1,

  # CRITICAL: NO NORMALIZATION - Python will handle this consistently
  normalize_rna = FALSE,   # Changed from TRUE
  normalize_atac = FALSE,  # Changed from TRUE

  # Output settings - UPDATE THIS TO YOUR PATH
  output_dir = "/data/halimaakhter/TF_Rscript/scMultiom_data/",
  save_metadata = TRUE,
  verbose = TRUE
)

# Utility functions
log_message <- function(message, verbose = CONFIG$verbose) {
  if (verbose) {
    cat(paste0("[", Sys.time(), "] ", message, "\n"))
  }
}

process_multiomics_data <- function(input_path) {
  dataset_name <- basename(input_path)
  log_message(sprintf("Processing dataset: %s", dataset_name))

  # Construct file paths
  h5_file <- file.path(input_path, "outs", "filtered_feature_bc_matrix.h5")
  barcode_metrics_file <- file.path(input_path, "outs", "per_barcode_metrics.csv")

  # Check if files exist
  log_message(sprintf("Checking H5 file: %s", h5_file))
  if (!file.exists(h5_file)) {
    stop(sprintf("Required file not found: %s", h5_file))
  }

  # Load 10x data
  log_message("Loading 10x multiomics data...")
  counts <- Read10X_h5(h5_file)

  # DEBUG: Print data structure
  log_message("=== DEBUG: DATA STRUCTURE ===")
  if (is.list(counts)) {
    log_message(sprintf("Available modalities: %s", paste(names(counts), collapse = ", ")))
    for (modality in names(counts)) {
      log_message(sprintf("%s dimensions: %d features x %d cells",
                          modality, nrow(counts[[modality]]), ncol(counts[[modality]])))
    }
  }

  # Check if multiomics data
  if (!is.list(counts) || !all(c("Gene Expression", "Peaks") %in% names(counts))) {
    stop("Input data does not appear to be multiomics (missing Gene Expression or Peaks)")
  }

  # Load barcode metrics if available
  if (file.exists(barcode_metrics_file)) {
    log_message("=== USING PER_BARCODE_METRICS.CSV ===")
    barcode_metrics <- read.csv(barcode_metrics_file, stringsAsFactors = FALSE)
    log_message(sprintf("Barcode metrics file loaded: %d rows, %d columns",
                        nrow(barcode_metrics), ncol(barcode_metrics)))

    # Filter for cells marked as high quality
    if ("is_cell" %in% colnames(barcode_metrics)) {
      quality_cells <- barcode_metrics[barcode_metrics$is_cell == 1, ]
      log_message(sprintf("High-quality cells (is_cell==1): %d out of %d",
                          nrow(quality_cells), nrow(barcode_metrics)))

      # Get valid barcodes
      if ("gex_barcode" %in% colnames(quality_cells) && "atac_barcode" %in% colnames(quality_cells)) {
        valid_gex_barcodes <- quality_cells$gex_barcode
        valid_atac_barcodes <- quality_cells$atac_barcode

        # Find intersection with actual data
        available_gex_cells <- intersect(valid_gex_barcodes, colnames(counts[["Gene Expression"]]))
        available_atac_cells <- intersect(valid_atac_barcodes, colnames(counts[["Peaks"]]))

        log_message(sprintf("High-quality cells in H5 - GEX: %d, ATAC: %d",
                            length(available_gex_cells), length(available_atac_cells)))

        # Use filtered cells if available
        if (length(available_gex_cells) > 0 && length(available_atac_cells) > 0) {
          counts[["Gene Expression"]] <- counts[["Gene Expression"]][, available_gex_cells, drop = FALSE]
          counts[["Peaks"]] <- counts[["Peaks"]][, available_atac_cells, drop = FALSE]
          log_message("Applied barcode metrics filtering")
        }
      }
    }
  } else {
    log_message("=== NO PER_BARCODE_METRICS.CSV FOUND - PROCEEDING WITHOUT ===")
  }

  # Create Seurat object with minimal filtering
  log_message("Creating Seurat object...")
  seurat_obj <- CreateSeuratObject(
    counts = counts[["Gene Expression"]],
    project = dataset_name,
    min.cells = CONFIG$min_cells_per_gene,
    min.features = CONFIG$min_genes_per_cell
  )

  log_message(sprintf("After CreateSeuratObject: %d genes x %d cells", nrow(seurat_obj), ncol(seurat_obj)))

  # Add ATAC data
  log_message("Adding ATAC-seq data...")

  # Find common cells between RNA and ATAC
  common_cells <- intersect(colnames(seurat_obj), colnames(counts[["Peaks"]]))
  log_message(sprintf("Common cells between RNA and ATAC: %d", length(common_cells)))

  if (length(common_cells) > 0) {
    # Subset to common cells
    seurat_obj <- subset(seurat_obj, cells = common_cells)
    atac_counts_subset <- counts[["Peaks"]][, common_cells, drop = FALSE]

    # Create ChromatinAssay
    atac_assay <- CreateChromatinAssay(
      counts = atac_counts_subset,
      sep = c(":", "-"),
      min.cells = CONFIG$min_peak_cells
    )
    seurat_obj[["ATAC"]] <- atac_assay

    log_message(sprintf("Final object: %d cells, %d genes, %d peaks",
                        ncol(seurat_obj), nrow(seurat_obj[["RNA"]]), nrow(seurat_obj[["ATAC"]])))
  } else {
    stop("No common cells found between RNA and ATAC data")
  }

  # Calculate basic QC metrics
  log_message("Calculating QC metrics...")
  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")

  # SKIP NORMALIZATION - will be done consistently in Python
  log_message("SKIPPING NORMALIZATION - will be done in Python with log1p")

  return(seurat_obj)
}

save_processed_data <- function(seurat_obj, dataset_name) {
  # Create output directory
  if (!dir.exists(CONFIG$output_dir)) {
    dir.create(CONFIG$output_dir, recursive = TRUE)
    log_message(sprintf("Created output directory: %s", CONFIG$output_dir))
  }

  # CRITICAL CHANGE: Extract RAW counts from "counts" slot, NOT "data" slot
  log_message("=== EXTRACTING RAW COUNTS (NOT NORMALIZED) ===")
  rna_counts <- GetAssayData(seurat_obj, assay = "RNA", slot = "counts")   # RAW counts
  atac_counts <- GetAssayData(seurat_obj, assay = "ATAC", slot = "counts") # RAW counts

  log_message(sprintf("RNA RAW counts dimensions: %d genes x %d cells", nrow(rna_counts), ncol(rna_counts)))
  log_message(sprintf("ATAC RAW counts dimensions: %d peaks x %d cells", nrow(atac_counts), ncol(atac_counts)))

  # Verify we have raw counts (should be integers)
  log_message(sprintf("RNA data type check - Max: %.2f, Has decimals: %s",
                      max(rna_counts), any(rna_counts != floor(rna_counts))))
  log_message(sprintf("ATAC data type check - Max: %.2f, Has decimals: %s",
                      max(atac_counts), any(atac_counts != floor(atac_counts))))

  # Convert to data frames with proper formatting
  log_message("Preparing RNA RAW data for export...")
  rna_df <- as.data.frame(as.matrix(t(rna_counts)))
  rna_df <- rownames_to_column(rna_df, var = "cell_id")
  log_message(sprintf("RNA dataframe: %d rows x %d columns", nrow(rna_df), ncol(rna_df)))

  log_message("Preparing ATAC RAW data for export...")
  atac_df <- as.data.frame(as.matrix(t(atac_counts)))
  atac_df <- rownames_to_column(atac_df, var = "cell_id")
  log_message(sprintf("ATAC dataframe: %d rows x %d columns", nrow(atac_df), ncol(atac_df)))

  # Define output file paths
  gex_file <- file.path(CONFIG$output_dir, paste0(dataset_name, "_GEX_RAW.csv"))
  peak_file <- file.path(CONFIG$output_dir, paste0(dataset_name, "_PEAK_RAW.csv"))

  # Save data
  log_message(sprintf("=== SAVING RAW COUNT FILES ==="))
  log_message(sprintf("Saving RNA RAW counts to: %s", gex_file))
  write.csv(rna_df, gex_file, row.names = FALSE)
  log_message(sprintf("RNA file saved successfully. Size: %s bytes", file.size(gex_file)))

  log_message(sprintf("Saving ATAC RAW counts to: %s", peak_file))
  write.csv(atac_df, peak_file, row.names = FALSE)
  log_message(sprintf("ATAC file saved successfully. Size: %s bytes", file.size(peak_file)))

  # Save metadata if requested
  if (CONFIG$save_metadata) {
    metadata_file <- file.path(CONFIG$output_dir, paste0(dataset_name, "_metadata.csv"))
    metadata_df <- seurat_obj@meta.data
    metadata_df <- rownames_to_column(metadata_df, var = "cell_id")

    log_message(sprintf("Saving metadata to: %s", metadata_file))
    write.csv(metadata_df, metadata_file, row.names = FALSE)
  }

  # Return summary statistics
  summary_stats <- list(
    dataset = dataset_name,
    n_cells = ncol(seurat_obj),
    n_genes = nrow(seurat_obj[["RNA"]]),
    n_peaks = nrow(seurat_obj[["ATAC"]]),
    mean_genes_per_cell = mean(seurat_obj$nFeature_RNA),
    mean_umi_per_cell = mean(seurat_obj$nCount_RNA),
    mean_mt_percent = mean(seurat_obj$percent.mt),
    files_created = c(gex_file, peak_file, if(CONFIG$save_metadata) metadata_file else NULL)
  )

  return(summary_stats)
}

# Main processing function
process_multiomics_datasets <- function(input_paths) {
  log_message("Starting RAW COUNT extraction pipeline...")
  log_message("NO NORMALIZATION will be applied - Python will handle this")
  log_message(sprintf("Output directory: %s", CONFIG$output_dir))

  all_summaries <- list()

  for (i in seq_along(input_paths)) {
    input_path <- input_paths[i]
    dataset_name <- basename(input_path)

    log_message(sprintf("\n=== Processing dataset %d/%d: %s ===",
                        i, length(input_paths), dataset_name))

    tryCatch({
      # Process the data
      seurat_obj <- process_multiomics_data(input_path)

      # Save the processed data
      summary_stats <- save_processed_data(seurat_obj, dataset_name)
      all_summaries[[dataset_name]] <- summary_stats

      log_message(sprintf("Successfully processed %s: %d cells, %d genes, %d peaks",
                          dataset_name, summary_stats$n_cells,
                          summary_stats$n_genes, summary_stats$n_peaks))

    }, error = function(e) {
      log_message(sprintf("ERROR processing %s: %s", dataset_name, e$message))
      traceback()
    })
  }

  # Save overall summary
  if (length(all_summaries) > 0) {
    summary_file <- file.path(CONFIG$output_dir, "processing_summary.csv")
    summary_df <- do.call(rbind, lapply(all_summaries, function(x) {
      data.frame(
        dataset = x$dataset,
        n_cells = x$n_cells,
        n_genes = x$n_genes,
        n_peaks = x$n_peaks,
        mean_genes_per_cell = round(x$mean_genes_per_cell, 2),
        mean_umi_per_cell = round(x$mean_umi_per_cell, 2),
        mean_mt_percent = round(x$mean_mt_percent, 2)
      )
    }))

    write.csv(summary_df, summary_file, row.names = FALSE)
    log_message(sprintf("\nProcessing complete! Summary saved to: %s", summary_file))
  }

  return(all_summaries)
}

# MAIN EXECUTION
# UPDATE THESE PATHS TO YOUR MULTIOMICS DATA LOCATION


input_paths <- c(
  "/destiny/halima/Marlin/4_10_2025_evryting_1.1T/Thesis_2/1_GD428_21136_Hu_REH_Parental/",
  
  "/destiny/halima/Marlin/4_10_2025_evryting_1.1T/Thesis_2/2_GD444_21136_Hu_Sup_Parental/"
  
  #"/destiny/halima/Marlin/4_10_2025_evryting_1.1T/Thesis_2/3_GD460_21136Hazlehurst_PC9/"
)

# Process all datasets
results <- process_multiomics_datasets(input_paths)

# Print final summary
cat("\n=== FINAL SUMMARY ===\n")
for (dataset_name in names(results)) {
  summary <- results[[dataset_name]]
  cat(sprintf("%s: %d cells, %d genes, %d peaks\n",
              dataset_name, summary$n_cells, summary$n_genes, summary$n_peaks))
}

cat("\n=== IMPORTANT ===\n")
cat("Files contain RAW COUNTS (integers)\n")
cat("Python will apply log1p(CPM) normalization consistently for both RNA and PEAK\n")
