# OpenMP runtime conflict detection and core resolution.
#
# On some platforms (notably macOS) the R session and Python's numerical stack
# can each load their own copy of an OpenMP runtime. Running multi-threaded code
# from both runtimes in the same process can corrupt thread state and crash the
# session. These helpers detect when an offending namespace has been loaded and
# clamp cores to a single thread for backends that use OpenMP.

# Session-scoped state used for the "once" warning cadence.
.fastpca_state <- new.env(parent = emptyenv())

# Packages known to map an OpenMP runtime that can collide with the one used by
# Python's numerical stack. Kept as data so future offenders can be added either
# here or, without editing this file, through the package option.
.openmp_offender_packages <- function() {
  offenders <- getOption("FastPCA.openmp_offenders",
                         default = c("data.table", "RcppArmadillo"))
  offenders[!is.na(offenders)]
}

# Return the offending packages that are currently loaded in this session.
.openmp_conflicts <- function() {
  intersect(.openmp_offender_packages(), loadedNamespaces())
}

# Backends whose native code can spawn OpenMP threads.
.backend_uses_openmp <- function(backend) {
  backend %in% c("pytorch", "rtorch")
}

.openmp_warning_mode <- function() {
  mode <- getOption("FastPCA.openmp_warning", default = "always")
  if (!is.character(mode) || length(mode) != 1L ||
      !mode %in% c("always", "once", "silent")) {
    mode <- "always"
  }
  mode
}

# Force the process-wide OpenMP thread pool to a single thread.
#
# This must run *before* an OpenMP-backed library (torch) is imported: torch
# builds its thread pool at import time, so clamping afterwards through
# torch.set_num_threads() is too late and the process can still segfault.
.cap_openmp_threads <- function(threads = 1L) {
  Sys.setenv(OMP_NUM_THREADS = as.character(threads))
  invisible(as.integer(threads))
}

# Emit the conflict warning according to FastPCA.openmp_warning.
.warn_openmp_conflict <- function(offenders) {
  mode <- .openmp_warning_mode()
  if (identical(mode, "silent")) return(invisible(NULL))
  if (identical(mode, "once") && isTRUE(.fastpca_state$warned)) return(invisible(NULL))
  .fastpca_state$warned <- TRUE
  warning(
    "FastPCA detected an OpenMP runtime conflict in this session (loaded: ",
    paste(offenders, collapse = ", "),
    "). To avoid a session crash, cores has been clamped to 1 for this call. ",
    "Restart R before loading these packages, or pass cores = 1 explicitly. ",
    "Set options(FastPCA.openmp_warning = 'once') or 'silent' to change ",
    "how often this is reported.",
    call. = FALSE
  )
}

# Validate a user-supplied cores and clamp it to 1 when running OpenMP-backed
# code would collide with an already-loaded foreign runtime. Returns a single
# positive integer.
.resolve_cores <- function(cores, openmp = TRUE,
                           offenders = .openmp_conflicts()) {
  cores <- suppressWarnings(as.integer(cores))
  if (length(cores) != 1L || is.na(cores) || cores < 1L) {
    stop("'cores' must be a single integer greater than or equal to 1.",
         call. = FALSE)
  }
  if (isTRUE(openmp) && length(offenders)) {
    # cap the thread pool before any OpenMP-backed library is imported, even
    # when the caller already asked for a single core.
    .cap_openmp_threads()
  }
  if (isTRUE(openmp) && cores > 1L && length(offenders)) {
    .warn_openmp_conflict(offenders)
    cores <- 1L
  }
  cores
}
