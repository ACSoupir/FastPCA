# NIPALS on the native R torch backend ---------------------------------------
#
# Mirrors .nipals_r() but keeps the working matrix on a torch device and uses
# tensor matvecs for the two products per iteration. Missing entries are held at
# zero and re-zeroed after every deflation by multiplying with the observation
# mask, so no masked value can leak into a numerator.
#
# The per-vector work (Gram-Schmidt, normalisation, convergence) stays in R:
# those vectors are length nobs or nvar and cost nothing next to the matvec.

.nipals_rtorch <- function(X, ncomp, tol, maxit, gramschmidt = TRUE,
                           device = "CPU", cores = 1L, verbose = FALSE) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("backend = 'rtorch' requires the 'torch' package; install it with ",
         "install.packages('torch').", call. = FALSE)
  }

  torch::torch_set_num_threads(as.integer(cores))
  if (identical(toupper(as.character(device)), "GPU") && torch::cuda_is_available()) {
    dev <- torch::torch_device("cuda")
  } else {
    if (identical(toupper(as.character(device)), "GPU")) {
      warning("CUDA supplied as device but not available to torch", call. = FALSE)
      message("Falling back to CPU for computation")
    }
    dev <- torch::torch_device("cpu")
  }

  nobs <- nrow(X)
  nvar <- ncol(X)
  na_mask <- is.na(X)
  has_na <- any(na_mask)

  Xz <- X
  if (has_na) Xz[na_mask] <- 0   # masked entries contribute zero to every product
  Z_t <- torch::torch_tensor(Xz, dtype = torch::torch_double())$to(device = dev)
  obs_t <- NULL
  empty_rows <- rep(FALSE, nobs)
  if (has_na) {
    obs_t <- torch::torch_tensor(matrix(as.numeric(!na_mask), nobs, nvar),
                                 dtype = torch::torch_double())$to(device = dev)
    empty_rows <- rowSums(!na_mask) == 0L
  }

  eig <- numeric(ncomp)
  loadings <- matrix(0, nvar, ncomp,
                     dimnames = list(colnames(X), .pc_names(ncomp)))
  scores <- matrix(0, nobs, ncomp,
                   dimnames = list(rownames(X), .pc_names(ncomp)))
  iterations <- integer(ncomp)
  converged <- logical(ncomp)

  for (h in seq_len(ncomp)) {
    colscore <- as.numeric(torch::torch_sum(Z_t$abs(), dim = 1L))
    scol <- which.max(colscore)
    th_t <- Z_t[, scol]
    th <- as.numeric(th_t)

    pciter <- 1L
    conv <- FALSE
    repeat {
      if (has_na) {
        denom_p_t <- torch::torch_matmul(obs_t$t(), th_t * th_t)
        denom_p <- as.numeric(denom_p_t)
        denom_p[denom_p == 0] <- .Machine$double.xmin
        ph <- as.numeric(torch::torch_matmul(Z_t$t(), th_t)) / denom_p
      } else {
        ph <- as.numeric(torch::torch_matmul(Z_t$t(), th_t)) / sum(th * th)
      }
      if (gramschmidt && h > 1L) {
        P <- loadings[, seq_len(h - 1L), drop = FALSE]
        # as.vector: %*% would return a one-column matrix, which torch turns
        # into a rank-2 tensor and the matvecs below then fail on shapes
        ph <- as.vector(ph - P %*% crossprod(P, ph))
      }
      nrm <- sqrt(sum(ph * ph))
      if (!is.finite(nrm) || nrm == 0) {
        stop(sprintf(paste0("NIPALS failed at component %d: the loading vector ",
                            "has zero norm; the data may be rank-deficient."), h),
             call. = FALSE)
      }
      ph <- ph / nrm

      th_old <- th
      ph_t <- torch::torch_tensor(ph, dtype = torch::torch_double())$to(device = dev)
      if (has_na) {
        denom_t <- as.numeric(torch::torch_matmul(obs_t, ph_t * ph_t))
        denom_t[denom_t == 0] <- .Machine$double.xmin
        th <- as.numeric(torch::torch_matmul(Z_t, ph_t)) / denom_t
      } else {
        th <- as.numeric(torch::torch_matmul(Z_t, ph_t)) / sum(ph * ph)
      }
      if (has_na) th[empty_rows] <- 0
      if (gramschmidt && h > 1L) {
        Tprev <- scores[, seq_len(h - 1L), drop = FALSE]
        sv_prev <- sqrt(eig[seq_len(h - 1L)])
        Tn <- sweep(Tprev, 2L, sv_prev, "/")
        th <- as.vector(th - Tn %*% crossprod(Tn, th))
      }

      nn <- sqrt(sum(th * th))
      no <- sqrt(sum(th_old * th_old))
      if (nn > 0 && no > 0) {
        d <- th / nn - th_old / no
        if (sum(d * d) < tol) {
          conv <- TRUE
          break
        }
      }
      if (pciter >= maxit) {
        warning(sprintf(paste0("NIPALS: component %d stopped after %d iterations ",
                               "without reaching tol = %g."), h, maxit, tol),
                call. = FALSE)
        break
      }
      pciter <- pciter + 1L

      # the next iteration needs th as a tensor again
      th_t <- torch::torch_tensor(th, dtype = torch::torch_double())$to(device = dev)
    }

    # rank-one deflation, then re-zero the masked entries
    th_t <- torch::torch_tensor(th, dtype = torch::torch_double())$to(device = dev)
    outer_t <- torch::torch_matmul(th_t$unsqueeze(2L), ph_t$unsqueeze(1L))
    Z_t <- Z_t - outer_t
    if (has_na) Z_t <- Z_t * obs_t

    loadings[, h] <- ph
    scores[, h] <- th
    eig[h] <- sum(th * th)
    iterations[h] <- pciter
    converged[h] <- conv

    if (verbose) {
      message(sprintf("  PC%d: %d iterations, eigenvalue %.4f", h, pciter, eig[h]))
    }
  }

  if (has_na && any(empty_rows)) scores[empty_rows, ] <- NA_real_

  list(eig = eig, loadings = loadings, scores = scores,
       iterations = iterations, converged = converged)
}
