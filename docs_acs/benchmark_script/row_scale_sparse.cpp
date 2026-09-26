// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat row_scale_sparse(const S4& mat) {
  IntegerVector dims = mat.slot("Dim");
  int nrow = dims[0];
  int ncol = dims[1];
  
  // Extract slots
  NumericVector x = mat.slot("x");
  IntegerVector i = mat.slot("i");    // row indices of nonzeros
  IntegerVector p = mat.slot("p");    // pointers to column starts
  
  // Sum and sumsq for each row
  arma::vec row_sum(nrow, arma::fill::zeros);
  arma::vec row_sumsq(nrow, arma::fill::zeros);
  
  // First pass: sum/sumsq for each row
  int nnz = x.size();
  for (int col = 0; col < ncol; ++col) {
    for (int idx = p[col]; idx < p[col+1]; ++idx) {
      int row = i[idx];
      double val = x[idx];
      row_sum[row] += val;
      row_sumsq[row] += val * val;
    }
  }
  
  // Compute mean and sd for each row (including zeros)
  arma::vec row_mean = row_sum / ncol;
  arma::vec row_var = (row_sumsq + (ncol - row_sum / row_mean) * 0) / ncol - arma::square(row_mean);
  // More numerically stable version:
  arma::vec row_sd(nrow);
  for (int r = 0; r < nrow; ++r) {
    double mean = row_mean[r];
    double ssq = row_sumsq[r] + (ncol - (row_sum[r] == 0 ? 0 : row_sum[r]/mean)) * 0.0;
    double var = ssq / ncol - mean * mean;
    row_sd[r] = std::sqrt(var);
    if (row_sd[r] == 0) row_sd[r] = 1.0; // avoid div by zero
  }
  
  // Second pass: fill dense result
  arma::mat result(nrow, ncol, arma::fill::zeros);
  
  // Fill nonzeros first
  for (int col = 0; col < ncol; ++col) {
    for (int idx = p[col]; idx < p[col+1]; ++idx) {
      int row = i[idx];
      double val = x[idx];
      result(row, col) = (val - row_mean[row]) / row_sd[row];
    }
  }
  // Fill zeros (optional, but already zero by arma::fill::zeros, so just scale)
  for (int row = 0; row < nrow; ++row) {
    double z = (0 - row_mean[row]) / row_sd[row];
    for (int col = 0; col < ncol; ++col) {
      // Only fill zeros
      if (result(row, col) == 0) result(row, col) = z;
    }
  }
  
  return result.t();
}
