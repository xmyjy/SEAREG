#!/usr/bin/env Rscript

# ===================== Motif annotation with JASPAR2024 =====================
# Inputs:  BED (peaks), reference genome FASTA (for bedtools getfasta)
# Outputs: freq_kmer.h5 (optional), freq_motif.h5
# Deps:    rhdf5, HDF5Array, Biostrings, Matrix, TFBSTools, JASPAR2024,
#          motifmatchr, SummarizedExperiment, DBI, RSQLite, optparse, (optional) BiocParallel

suppressMessages(library(optparse, quietly = TRUE))
suppressMessages(library(methods))

# ----------------------------- helpers -----------------------------
as_dgC <- function(M) {
  if (inherits(M, "dgCMatrix")) return(M)
  Matrix::as(Matrix::Matrix(M, sparse = TRUE), "dgCMatrix")
}

drop0_dgC <- function(M) {
  M <- as_dgC(M)
  Matrix::drop0(M)
}

transform_nonzero <- function(M, how = c("none", "log1p"), clamp_nonneg = FALSE) {
  how <- match.arg(how)
  M <- as_dgC(M)
  if (clamp_nonneg) M@x <- pmax(0, M@x)
  if (how == "log1p") M@x <- log1p(M@x)
  M
}

topk_per_col <- function(M, K = 0L) {
  M <- as_dgC(M)
  if (is.null(K) || K <= 0) return(M)

  p <- M@p; i <- M@i; x <- M@x
  nc <- ncol(M)

  ii <- vector("list", nc)
  jj <- vector("list", nc)
  xx <- vector("list", nc)

  for (j in seq_len(nc)) {
    start <- p[j] + 1L
    end   <- p[j + 1L]
    if (start > end) next

    idx <- start:end
    if (length(idx) > K) {
      ord <- order(x[idx], decreasing = TRUE)[seq_len(K)]
      idx <- idx[ord]
    }
    ii[[j]] <- i[idx] + 1L
    jj[[j]] <- rep.int(j, length(idx))
    xx[[j]] <- x[idx]
  }

  i_new <- unlist(ii, use.names = FALSE)
  j_new <- unlist(jj, use.names = FALSE)
  x_new <- unlist(xx, use.names = FALSE)

  M2 <- Matrix::sparseMatrix(
    i = i_new, j = j_new, x = x_new,
    dims = dim(M), giveCsparse = TRUE
  )
  dimnames(M2) <- dimnames(M)   # ✅关键：把行列名带回去
  Matrix::drop0(M2)
}


topk_per_row <- function(M, K = 0L) {
  M <- as_dgC(M)
  if (is.null(K) || K <= 0) return(M)

  Mt <- Matrix::t(M)
  Mt <- as_dgC(Mt)
  Mt2 <- topk_per_col(Mt, K = K)
  M2 <- Matrix::t(Mt2)
  M2 <- as_dgC(M2)
  dimnames(M2) <- dimnames(M)
  Matrix::drop0(M2)
}


apply_idf_col <- function(M) {
  M <- as_dgC(M)
  df <- diff(M@p)
  N <- nrow(M)
  idf <- log((N + 1) / (df + 1))  # length = ncol

  # multiply each column's nonzeros by idf
  M@x <- M@x * rep(idf, times = df)
  M
}

# ----------------------------- main -----------------------------
main <- function() {
  option_list <- list(
    make_option(c("-i","--input"), type="character", help="Input peaks .bed"),
    make_option(c("-g","--genome"), type="character", help="Reference genome FASTA (for bedtools getfasta)"),
    make_option(c("--no_kmer"),  action="store_true", default=FALSE, help="Disable k-mer feature extraction"),
    make_option(c("--no_motif"), action="store_true", default=FALSE, help="Disable motif scanning"),
    make_option(c("-k","--k_kmer"), type="integer", default=6, help="k for k-mer (default: 6)"),
    make_option(c("--tax_group"), type="character", default=NULL, help="JASPAR tax_group (e.g. vertebrates)"),
    make_option(c("--species_names"), type="character", default=NULL, help="Comma-separated species (e.g. 'Homo sapiens,Mus musculus')"),
    make_option(c("--collection"), type="character", default="CORE", help="JASPAR collection (default: CORE)"),
    make_option(c("--min_motifs"), type="integer", default=300, help="Minimum PFMs to accept before fallback"),
    make_option(c("--p_cutoff"), type="double",  default=5e-5, help="motifmatchr p-value cutoff (default: 5e-5)"),
    make_option(c("--mode"), type="character", default="scores", help="Motif mode: matches|positions|scores"),
    make_option(c("--scores_assay"), type="character", default="weighted",
                help="When --mode scores: which matrix to export: scores|counts|matches|weighted"),
    make_option(c("--score_transform"), type="character", default="log1p",
                help="Transform motifScores nonzeros: none|log1p"),
    make_option(c("--count_transform"), type="character", default="log1p",
                help="Transform motifCounts nonzeros: none|log1p"),
    make_option(c("--weighted_use_counts"), action="store_true", default=TRUE,
                help="When scores_assay=weighted, multiply scores by (1+transformed counts)"),
    make_option(c("--topk_frac_per_motif"), type="double", default=0.04,
                help="Keep top fraction of peaks per motif column (used when --topk_per_motif=0). e.g. 0.04"),

    make_option(c("--topk_min_per_motif"), type="integer", default=500,
                help="Min K when using --topk_frac_per_motif"),
    make_option(c("--topk_max_per_motif"), type="integer", default=5000,
                help="Max K when using --topk_frac_per_motif"),

    make_option(c("--topk_per_peak"), type="integer", default=30,
            help="Keep topK motifs per peak row (0 disables). e.g. 15/20"),
    make_option(c("--apply_idf"), action="store_true", default=FALSE,
                help="Apply IDF scaling per motif: log((N+1)/(df+1))"),
    make_option(c("--scale_method"), type="character", default="none",
                help="Scaling per motif column (nonzeros): none|zscore|quantile"),
    make_option(c("--zscore_post"), type="character", default="sigmoid",
                help="Post-processing after zscore: sigmoid|none|relu (to keep weights non-negative)"),
    make_option(c("--quantile_to_normal"), action="store_true", default=FALSE,
                help="If scale_method=quantile, map quantile to N(0,1) via qnorm (may produce negatives)"),
    make_option(c("--max_df_frac"), type="double", default=0.3,
                help="Drop motifs appearing in > max_df_frac of peaks (set >=1 to disable)"),
    make_option(c("--min_peaks"), type="integer", default=20,
                help="Drop motifs appearing in < min_peaks peaks (set <=0 to disable)"),
    make_option(c("--batch_motifs"), type="integer", default=150, help="PFM batch size"),
    make_option(c("--threads"), type="integer", default=1, help="Threads for MOODS/BiocParallel"),
    make_option(c("-o","--output"), type="character", default="output_kmers_motifs", help="Output directory")
  )
  opt <- parse_args(OptionParser(option_list = option_list))

  if (is.null(opt$input)) stop("Please provide --input BED")
  if (!opt$no_motif && is.null(opt$genome)) stop("When motifs enabled, provide --genome FASTA")

  infile_bed   <- opt$input
  genome_fa    <- opt$genome
  outdir       <- opt$output
  k            <- opt$k_kmer
  threads      <- max(1L, opt$threads)

  threads <- max(1L, opt$threads)
  if (threads > 1) {
        # 多进程模式下，底层单线程
      Sys.setenv(OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1)
  } else {
      Sys.setenv(OMP_NUM_THREADS=1)
  }
  tax_group    <- opt$tax_group
  species_str  <- opt$species_names
  collection   <- opt$collection
  min_motifs   <- opt$min_motifs
  p_cutoff     <- opt$p_cutoff
  mode         <- tolower(opt$mode)
  if (!mode %in% c("matches","positions","scores")) stop("--mode must be matches|positions|scores")
  scores_assay <- tolower(opt$scores_assay)
  if (!scores_assay %in% c("scores","counts","matches","weighted")) stop("--scores_assay must be scores|counts|matches|weighted")

  score_transform <- tolower(opt$score_transform)
  if (!score_transform %in% c("none","log1p")) stop("--score_transform must be none|log1p")
  count_transform <- tolower(opt$count_transform)
  if (!count_transform %in% c("none","log1p")) stop("--count_transform must be none|log1p")

  scale_method <- tolower(opt$scale_method)
  if (!scale_method %in% c("none","zscore","quantile")) stop("--scale_method must be none|zscore|quantile")
  zscore_post <- tolower(opt$zscore_post)
  if (!zscore_post %in% c("sigmoid","none","relu")) stop("--zscore_post must be sigmoid|none|relu")

  batch_motifs <- max(1L, opt$batch_motifs)

  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)


  suppressMessages({
    library(rhdf5); library(HDF5Array); library(Biostrings); library(Matrix)
    library(DBI); library(RSQLite)
    library(TFBSTools); library(JASPAR2024); library(motifmatchr); library(SummarizedExperiment)
  })

  # ------------------------------- BED -> FASTA -------------------------------
  peaks_seq <- NULL; peaks_name <- NULL
  if (!opt$no_kmer || !opt$no_motif) {
    message("Converting BED -> FASTA via bedtools ...")
    fasta_out <- file.path(outdir, paste0(basename(infile_bed), ".fa"))
    cmd <- sprintf("bedtools getfasta -fi %s -bed %s -fo %s",
                   shQuote(genome_fa), shQuote(infile_bed), shQuote(fasta_out))
    if (system(cmd) != 0) stop("bedtools getfasta failed.")
    peaks_seq  <- readDNAStringSet(fasta_out, "fasta")
    peaks_name <- gsub(":|-","_", names(peaks_seq))  # chr_start_end
  }

  # --------------------------------- k-mers ----------------------------------
  if (!opt$no_kmer) {
    message("Scanning k-mers (k=", k, ") ...")
    freq_k <- oligonucleotideFrequency(peaks_seq, k)
    rownames(freq_k) <- peaks_name
    freq_k <- as(freq_k, "sparseMatrix")
    kmer_out <- file.path(outdir, "freq_kmer.h5")
    writeHDF5Array(t(freq_k), kmer_out, name="mat", with.dimnames=FALSE, verbose=FALSE)
    rhdf5::h5write(rownames(freq_k), kmer_out, "row_names")
    rhdf5::h5write(colnames(freq_k), kmer_out, "col_names")
    message("Saved: ", kmer_out, "  (peaks x kmers = ", nrow(freq_k), " x ", ncol(freq_k), ")")
  }

  # ------------------------------ Fetch PFMs (2024) --------------------------
  fetch_pfms_2024 <- function(tax_group, species_str, collection="CORE", min_motifs=300) {
    obj <- JASPAR2024::JASPAR2024()
    db_path <- tryCatch(db(obj),
                        error=function(e){
                          get("db", envir=asNamespace("JASPAR2024"))(obj)
                        })
    con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
    on.exit(try(DBI::dbDisconnect(con), silent = TRUE), add = TRUE)

    # 容器：用于存放所有符合条件的 Motif
    pfm_list <- list()
    user_specified_any <- FALSE # 标记用户是否指定了筛选条件

    # 1. 获取 Tax Group (不立即返回，而是存入列表)
    if (!is.null(tax_group) && nzchar(tax_group)) {
      user_specified_any <- TRUE
      message("Fetching PFMs by tax_group='", tax_group, "' ...")
      res <- tryCatch(TFBSTools::getMatrixSet(con, opts=list(tax_group=tax_group, collection=collection)),
                      error=function(e) NULL)
      if (!is.null(res) && length(res) > 0) {
        pfm_list <- c(pfm_list, list(res))
      }
    }

    # 2. 获取 Species (不立即返回，累加到列表)
    if (!is.null(species_str) && nzchar(species_str)) {
      user_specified_any <- TRUE
      sp <- trimws(unlist(strsplit(species_str, ",")))
      for (s in sp) {
        message("Fetching PFMs by species='", s, "' ...")
        res <- tryCatch(TFBSTools::getMatrixSet(con, opts=list(species=s, collection=collection)),
                        error=function(e) NULL)
        if (!is.null(res) && length(res) > 0) {
          pfm_list <- c(pfm_list, list(res))
        }
      }
    }

    # 3. 合并与去重
    pfm <- NULL
    if (length(pfm_list) > 0) {
      # 合并所有列表
      pfm_combined <- do.call(c, pfm_list)

      # 根据 JASPAR ID 去重 (防止 species 包含在 tax_group 里导致重复)
      ids <- vapply(pfm_combined, TFBSTools::ID, character(1))
      pfm <- pfm_combined[!duplicated(ids)]

      message(sprintf("Collected %d unique PFMs from user criteria.", length(pfm)))
    }

    # 4. Fallback 逻辑 (数量不够时触发)
    current_count <- if (is.null(pfm)) 0 else length(pfm)

    if (current_count < min_motifs) {
      if (user_specified_any) {
        message(sprintf("Warning: User filters retrieved only %d motifs (min_motifs=%d).", current_count, min_motifs))
        message("Adding 'vertebrates' as fallback to ensure sufficient coverage...")
      } else {
        message("No tax_group or species provided. Using 'vertebrates' by default...")
      }

      fallback <- tryCatch(TFBSTools::getMatrixSet(con, opts=list(tax_group="vertebrates", collection=collection)),
                           error=function(e) NULL)

      if (!is.null(fallback)) {
        if (is.null(pfm)) {
          pfm <- fallback
        } else {
          # 将 Fallback 结果合并进来并再次去重
          pfm_combined <- c(pfm, fallback)
          ids <- vapply(pfm_combined, TFBSTools::ID, character(1))
          pfm <- pfm_combined[!duplicated(ids)]
        }
      }
    }

    # 最终报告
    if (is.null(pfm)) return(NULL) # 实在找不到
    message(sprintf("Final PFM count: %d", length(pfm)))
    return(pfm)
  }

  # --------------------------------- Motifs ----------------------------------
  if (!opt$no_motif) {
    message("Fetching JASPAR2024 PFMs (collection=", collection, ") ...")
    pfm <- fetch_pfms_2024(tax_group, species_str, collection, min_motifs)
    if (length(pfm) == 0) stop("No PFMs retrieved from JASPAR2024.")

    ids <- names(pfm)
    tf_names <- vapply(pfm, function(m) TFBSTools::name(m), character(1))
    nice_names <- make.unique(gsub("[^A-Za-z0-9_]+","_", paste0(tf_names, "_", ids)))
    names(pfm) <- nice_names

    # Parallel backend (optional)
    bp <- NULL; has_BPPARAM <- FALSE
    if (threads > 1) {
      suppressMessages(library(BiocParallel))
      if (.Platform$OS.type == "windows") {
        bp <- BiocParallel::SnowParam(workers=threads, type="SOCK", progressbar=FALSE)
      } else {
        bp <- BiocParallel::MulticoreParam(workers=threads, progressbar=FALSE)
      }
    }
    if ("BPPARAM" %in% names(formals(motifmatchr::matchMotifs))) has_BPPARAM <- TRUE

    # Batch scanning
    N <- length(pfm)
    batches <- split(seq_len(N), ceiling(seq_len(N)/batch_motifs))
    message(sprintf("Matching motifs (mode='%s', p_cutoff=%.1e, threads=%d, batches=%d, batch_size~%d) ...",
                    mode, p_cutoff, threads, length(batches), batch_motifs))

    res_list <- vector("list", length(batches))
    for (bi in seq_along(batches)) {
      idx <- batches[[bi]]
      pfm_chunk <- pfm[idx]
      message(sprintf("  [Batch %d/%d] motifs %d..%d", bi, length(batches), min(idx), max(idx)))

      out_arg <- mode  # matches|positions|scores
      mm <- tryCatch({
        if (!is.null(bp) && has_BPPARAM) {
          motifmatchr::matchMotifs(pfm_chunk, peaks_seq, out=out_arg, p.cutoff=p_cutoff, BPPARAM=bp)
        } else if (!is.null(bp) && !has_BPPARAM) {
          if (requireNamespace("BiocParallel", quietly=TRUE)) BiocParallel::register(bp)
          motifmatchr::matchMotifs(pfm_chunk, peaks_seq, out=out_arg, p.cutoff=p_cutoff)
        } else {
          motifmatchr::matchMotifs(pfm_chunk, peaks_seq, out=out_arg, p.cutoff=p_cutoff)
        }
      }, error=function(e) e)
      if (inherits(mm, "error")) stop("matchMotifs failed in batch ", bi, ": ", mm$message)

      # ---- choose exported matrix (peaks x motifs) ----
      if (mode == "matches") {
        lm <- motifmatchr::motifMatches(mm)
        M <- as_dgC(lm * 1L)
      } else if (mode == "positions") {
        mc <- motifmatchr::motifCounts(mm)
        M <- as_dgC(mc)
      } else { # scores
        if (scores_assay == "scores") {
          ms <- motifmatchr::motifScores(mm)
          M <- transform_nonzero(as_dgC(ms), score_transform, clamp_nonneg=TRUE)
        } else if (scores_assay == "counts") {
          mc <- motifmatchr::motifCounts(mm)
          M <- transform_nonzero(as_dgC(mc), count_transform)
        } else if (scores_assay == "matches") {
          lm <- motifmatchr::motifMatches(mm)
          M <- as_dgC(lm * 1L)
        } else if (scores_assay == "weighted") {
          ms <- transform_nonzero(as_dgC(motifmatchr::motifScores(mm)), score_transform, clamp_nonneg=TRUE)
          if (isTRUE(opt$weighted_use_counts)) {
            mc <- transform_nonzero(as_dgC(motifmatchr::motifCounts(mm)), count_transform)
            # counts weight: (1 + transformed_counts)
            mc@x <- 1 + mc@x
            M <- ms * mc
          } else {
            M <- ms
          }
        }
      }

      colnames(M) <- names(pfm_chunk)
      rownames(M) <- peaks_name
      res_list[[bi]] <- drop0_dgC(M)
      rm(mm); gc()
    }

    # Combine batches
    freq_motif <- res_list[[1]]
    if (length(res_list) > 1) freq_motif <- do.call(cbind, res_list)
    freq_motif <- drop0_dgC(freq_motif)
    message(sprintf("Raw combined: peaks x motifs = %d x %d, nnz=%d",
                    nrow(freq_motif), ncol(freq_motif), length(freq_motif@x)))

    # 1) df filter (max_df_frac/min_peaks)
    n_peaks <- nrow(freq_motif)
    df0 <- diff(freq_motif@p)
    max_limit <- if (!is.na(opt$max_df_frac) && opt$max_df_frac < 1) opt$max_df_frac * n_peaks else Inf
    min_limit <- if (!is.na(opt$min_peaks) && opt$min_peaks > 0) opt$min_peaks else 0

    keep_mask <- (df0 <= max_limit) & (df0 >= min_limit)
    n_dropped_high <- sum(df0 > max_limit)
    n_dropped_low  <- sum(df0 < min_limit)

    message(sprintf(">>> Filtering Motifs:"))
    message(sprintf("    Total Scanned: %d", ncol(freq_motif)))
    message(sprintf("    Dropped High-freq (>%s coverage): %d",
                    ifelse(is.finite(max_limit), sprintf("%.2f%%", 100*opt$max_df_frac), "disabled"), n_dropped_high))
    message(sprintf("    Dropped Low-freq  (<%d peaks):    %d", min_limit, n_dropped_low))

    freq_motif <- freq_motif[, keep_mask, drop=FALSE]
    freq_motif <- drop0_dgC(freq_motif)
    message(sprintf("    Remaining Motifs: %d, nnz=%d", ncol(freq_motif), length(freq_motif@x)))

    # 2) topK per motif (fixed K or proportional K)
    K_eff <- opt$topk_per_motif

    if (is.null(K_eff) || K_eff <= 0) {
      frac <- opt$topk_frac_per_motif
      if (!is.null(frac) && frac > 0) {
        K_eff <- as.integer(round(frac * nrow(freq_motif)))
        K_eff <- max(opt$topk_min_per_motif, min(K_eff, opt$topk_max_per_motif))
      } else {
        K_eff <- 0L
      }
    }

    if (K_eff > 0) {
      freq_motif <- topk_per_col(freq_motif, K = K_eff)
      message(sprintf("After topK(K=%d, frac=%.3f): nnz=%d",
                      K_eff, opt$topk_frac_per_motif, length(freq_motif@x)))
    }

    # 3) IDF
    if (isTRUE(opt$apply_idf)) {
      freq_motif <- apply_idf_col(freq_motif)
      freq_motif <- drop0_dgC(freq_motif)
      message("Applied IDF scaling per motif.")
    }

    # 4) topK per peak
    if (!is.null(opt$topk_per_peak) && opt$topk_per_peak > 0) {
      freq_motif <- topk_per_row(freq_motif, K = opt$topk_per_peak)
      message(sprintf("After topK_per_peak(%d): nnz=%d", opt$topk_per_peak, length(freq_motif@x)))
    }
    # Save
    motif_out <- file.path(outdir, "freq_motif.h5")
    writeHDF5Array(t(freq_motif), motif_out, name="mat", with.dimnames=FALSE, verbose=FALSE)
    rhdf5::h5write(rownames(freq_motif), motif_out, "row_names")
    rhdf5::h5write(colnames(freq_motif), motif_out, "col_names")
    message("Saved: ", motif_out, "  (peaks x motifs = ", nrow(freq_motif), " x ", ncol(freq_motif), ")")
  }

  sink(file.path(outdir, "sessionInfo.txt"))
  print(sessionInfo())
  sink()
  message("Finished.")
}

tryCatch(main(), error=function(e){ message("ERROR: ", e$message); quit(status=1) })
