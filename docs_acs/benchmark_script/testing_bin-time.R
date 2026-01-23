#/usr/bin/time -l Rscript --vanilla docs_acs/benchmark_script/testing_bin-time.R
source("docs_acs/benchmark_script/0.1.project_functions.R")
library(FastPCA)
setup_py_env()
start_FastPCA_env()
# gex = readRDS("data/gex.rds")
# gc(full=TRUE)
#
# gex_vec = as.numeric(gex)
# gex2 = matrix(gex_vec, nrow = nrow(gex))
# scaled_mat = prep_matrix(gex2+1, transpose = TRUE, backend = "pytorch") #add 1 because pesky 0s
# saveRDS(scaled_mat, "data/prepped_gex.rds")
scaled_mat = readRDS("data/prepped_gex.rds")

k = 100
oversampling = 10
power_iterations = 10

fastpca_res_c1 = FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch")
