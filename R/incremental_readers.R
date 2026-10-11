# Chunked source reading for IncrementalPCA() ----------------------------------
#
# The point of the incremental solver is that nothing larger than one block is
# ever resident, so reading and computing are split: every source file becomes an
# iterator over blocks. Built-in CSV sources stream through a single base R
# connection (one open handle, one block in memory at a time). A user-supplied
# \code{reader} is treated as whole-file and sliced in memory afterwards; that is
# the extension point for containers R cannot stream here (parquet, HDF5).

# Extensions handled without extra packages. "zst"/"zstd" needs the external
# \code{zstd} binary, which is checked at read time.
.chunk_plain_extensions <- c("", "csv", "txt")
.chunk_gzip_extensions <- c("gz", "gzip")

# One open connection per chunk file. Compression is detected from the
# extension, matching how read.csv()/read.table() are usually pointed at a path.
.open_chunk_connection <- function(path) {
  ext <- tolower(tools::file_ext(basename(path)))
  if (ext %in% .chunk_plain_extensions) {
    return(file(path, open = "r"))
  }
  if (ext %in% .chunk_gzip_extensions) {
    return(gzfile(path, open = "r"))
  }
  if (ext == "bz2") {
    return(bzfile(path, open = "r"))
  }
  if (ext == "xz") {
    return(xzfile(path, open = "r"))
  }
  if (ext %in% c("zst", "zstd")) {
    zstd <- Sys.which("zstd")
    if (!nzchar(zstd)) {
      stop("Reading '", basename(path), "' needs the external `zstd` ",
           "binary, which was not found on the PATH. Install zstd (for example ",
           "`brew install zstd`) or decompress the file first.",
           call. = FALSE)
    }
    return(pipe(paste(shQuote(unname(zstd)), "-dc", shQuote(path)), "r"))
  }
  stop("Unsupported chunk file type '.", ext, "' (file '", basename(path),
       "'). Supported: plain text/CSV, .gz, .bz2, .xz and (with the zstd ",
       "binary) .zst/.zstd. For other containers convert to CSV or pass a ",
       "`reader` function.", call. = FALSE)
}

# Turn one parsed data frame into the block contract: a double matrix plus
# optional observation identifiers. A non-numeric column is almost always an
# identifier column the caller forgot to declare, so it gets its own message.
.block_from_frame <- function(df, path) {
  if (ncol(df) < 1L || nrow(df) < 1L) return(NULL)
  non_numeric <- names(df)[!vapply(df, is.numeric, logical(1))]
  if (length(non_numeric) > 0L) {
    stop("Chunk file '", basename(path), "' has non-numeric column(s): ",
         paste(utils::head(non_numeric, 5L), collapse = ", "),
         ". If the first column holds observation identifiers, pass ",
         "`row.names = 1` (or the column name).", call. = FALSE)
  }
  ids <- rownames(df)
  # read.table() numbers the rows when row.names is not requested
  if (identical(ids, as.character(seq_len(nrow(df))))) ids <- NULL
  x <- as.matrix(df)
  storage.mode(x) <- "double"
  list(x = x, ids = ids)
}

# Convert whatever a custom reader returned into the same block contract.
.block_from_reader_output <- function(data, path) {
  if (is.data.frame(data)) return(.block_from_frame(data, path))
  if (!is.matrix(data) || !is.numeric(data)) {
    stop("`reader(path)` must return a numeric matrix or data frame for '",
         basename(path), "'.", call. = FALSE)
  }
  ids <- rownames(data)
  if (identical(ids, as.character(seq_len(nrow(data))))) ids <- NULL
  x <- data
  storage.mode(x) <- "double"
  list(x = x, ids = ids)
}

# A block source is an environment with next_block() and close(). The built-in
# reader keeps one connection open for the whole file; nrows is used so a single
# CSV larger than RAM never lands in memory at once.
.new_block_source <- function(path, header = TRUE, row.names = NULL,
                              sep = ",", chunk_rows = NULL, reader = NULL) {
  if (!is.null(reader)) {
    return(.new_whole_file_source(path, reader = reader,
                                  chunk_rows = chunk_rows))
  }

  con <- .open_chunk_connection(path)
  state <- new.env(parent = emptyenv())
  state$header_pending <- isTRUE(header)
  state$finished <- FALSE
  # a second block of the same file is read with header = FALSE, so the feature
  # names have to be carried over or every later block would look like V1, V2...
  state$names <- NULL

  state$next_block <- function() {
    if (isTRUE(state$finished)) return(NULL)
    nrows <- if (is.null(chunk_rows)) -1L else as.integer(chunk_rows)
    df <- tryCatch(
      utils::read.table(con, sep = sep,
                        header = isTRUE(state$header_pending),
                        row.names = row.names, check.names = FALSE,
                        stringsAsFactors = FALSE, quote = "\"",
                        comment.char = "", nrows = nrows),
      error = function(e) {
        # scan() reports an exhausted connection rather than returning 0 rows;
        # that is the normal end of a file whose last block was partial
        msg <- conditionMessage(e)
        if (grepl("no lines available in input", msg, fixed = TRUE)) {
          return(NULL)
        }
        stop("Could not read chunk file '", basename(path), "': ", msg,
             call. = FALSE)
      }
    )
    if (is.null(df)) {
      state$finished <- TRUE
      return(NULL)
    }
    # the header is consumed by read.table() on the first call only
    if (isTRUE(state$header_pending)) {
      state$names <- names(df)
      state$header_pending <- FALSE
    } else if (!is.null(state$names) && ncol(df) == length(state$names)) {
      names(df) <- state$names
    }
    block <- .block_from_frame(df, path)
    if (is.null(block)) {
      state$finished <- TRUE
      return(NULL)
    }
    if (is.null(chunk_rows)) state$finished <- TRUE
    block
  }
  state$close <- function() {
    try(close(con), silent = TRUE)
    invisible(NULL)
  }
  state
}

# Reader-supplied sources are sliced in memory: a custom reader is documented as
# returning the whole source, so block boundaries are drawn after it returns.
.new_whole_file_source <- function(path, reader, chunk_rows = NULL) {
  data <- tryCatch(reader(path), error = function(e) {
    stop("`reader()` failed for '", basename(path), "': ",
         conditionMessage(e), call. = FALSE)
  })
  block <- .block_from_reader_output(data, path)
  state <- new.env(parent = emptyenv())
  if (is.null(block)) {
    state$next_block <- function() NULL
    state$close <- function() invisible(NULL)
    return(state)
  }
  size <- if (is.null(chunk_rows)) nrow(block$x) else as.integer(chunk_rows)
  state$position <- 1L
  n_total <- nrow(block$x)

  state$next_block <- function() {
    if (state$position > n_total) return(NULL)
    rows <- state$position:min(state$position + size - 1L, n_total)
    state$position <- max(rows) + 1L
    list(x = block$x[rows, , drop = FALSE],
         ids = if (is.null(block$ids)) NULL else block$ids[rows])
  }
  state$close <- function() invisible(NULL)
  state
}

# Flatten the file list into one block iterator, checking feature alignment as
# it goes so a mismatch is reported with the offending file name.
.new_incremental_stream <- function(files, opts) {
  st <- new.env(parent = emptyenv())
  st$index <- 0L
  st$source <- NULL
  st$feature_names <- opts$feature_names
  st$n_features <- if (is.null(opts$feature_names)) NA_integer_ else
    length(opts$feature_names)
  st$n_rows <- 0L

  st$next_block <- function() {
    repeat {
      if (!is.null(st$source)) {
        block <- st$source$next_block()
        if (!is.null(block)) {
          out <- .align_incremental_block(block, st, files[[st$index]],
                                          file_index = st$index)
          if (is.null(out)) next
          st$n_rows <- st$n_rows + nrow(out$x)
          return(out)
        }
        st$source$close()
        st$source <- NULL
      }
      if (st$index >= length(files)) return(NULL)
      st$index <- st$index + 1L
      path <- files[[st$index]]
      if (!file.exists(path)) {
        stop("Chunk file does not exist: '", path, "'.", call. = FALSE)
      }
      st$source <- .new_block_source(path, header = opts$header,
                                     row.names = opts$row.names,
                                     sep = opts$sep,
                                     chunk_rows = opts$chunk_rows,
                                     reader = opts$reader)
    }
  }

  st$close <- function() {
    if (!is.null(st$source)) {
      st$source$close()
      st$source <- NULL
    }
    invisible(NULL)
  }
  st
}

# Align one block with the feature set of the first block. Returns NULL for a
# zero-row file so callers can skip it.
.align_incremental_block <- function(block, st, path, file_index = NA_integer_) {
  x <- block$x
  if (nrow(x) == 0L || ncol(x) == 0L) return(NULL)
  nm <- colnames(x)
  if (is.null(nm)) nm <- paste0("V", seq_len(ncol(x)))
  if (anyDuplicated(nm)) {
    stop("Chunk file '", basename(path), "' has duplicated feature names: ",
         paste(utils::head(unique(nm[duplicated(nm)]), 5L), collapse = ", "),
         ".", call. = FALSE)
  }
  colnames(x) <- nm

  if (is.na(st$n_features)) {
    st$feature_names <- nm
    st$n_features <- length(nm)
  } else if (length(nm) != st$n_features) {
    stop("Chunk file '", basename(path), "' has ", length(nm),
         " feature column(s) but earlier files have ", st$n_features,
         "; every chunk must share the same features.", call. = FALSE)
  } else if (!identical(nm, st$feature_names)) {
    missing <- setdiff(st$feature_names, nm)
    extra <- setdiff(nm, st$feature_names)
    if (length(missing) > 0L || length(extra) > 0L) {
      stop("Chunk file '", basename(path), "' has a different feature set; ",
           if (length(missing)) paste0("missing: ",
             paste(utils::head(missing, 5L), collapse = ", "),
             if (length(missing) > 5L) ", ..." else "") else "",
           if (length(extra)) paste0(" unexpected: ",
             paste(utils::head(extra, 5L), collapse = ", "),
             if (length(extra) > 5L) ", ..." else "") else "", ".",
           call. = FALSE)
    }
    x <- x[, st$feature_names, drop = FALSE]
  }
  list(x = x, ids = block$ids, file_index = file_index)
}
