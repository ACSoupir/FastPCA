#/usr/bin/time -l Rscript --vanilla docs_acs/benchmark_script/testing_bin-time.R
source("docs_acs/benchmark_script/0.1.project_functions.R")
library(FastPCA)
setup_py_env()
start_FastPCA_env()
gex = readRDS("data/gex.rds")
gc(full=TRUE)

gex_vec = as.numeric(gex)
gex2 = matrix(gex_vec, nrow = nrow(gex))
scaled_mat = prep_matrix(gex2+1, transpose = TRUE, backend = "pytorch") #add 1 because pesky 0s
saveRDS(scaled_mat, "data/prepped_gex.rds")

library(bigstatsr)
A = as_FBM(scaled_mat, type="double",
           backingfile = "data/prepped_gex_fbm")
A$save()
