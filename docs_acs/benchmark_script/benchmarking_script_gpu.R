#install.packages('bench', repos = 'https://cran.r-project.org')
#ran on L40 GPU
setwd("FastPCA")
source("0.1.project_functions.R")
library(Matrix)
library(FastPCA)
library(bench)
setup_py_env()
start_FastPCA_env()

#data
gex = readRDS("gex.rds")

gc(full=TRUE)
#also transposes now
#convert back to numeric becuase currently is int
gex_vec = as.numeric(gex)
gex2 = matrix(gex_vec, nrow = nrow(gex))
scaled_mat = prep_matrix(gex2+1, transpose = TRUE, backend = "pytorch") #add 1 because pesky 0s

k = 100
oversampling = 10
power_iterations = 10

fastpca_res_c1 = FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch", device = "GPU")
benchmark_results = bench::mark(
  fastpca_res_c1 = FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch", device = "GPU"),
  min_iterations = 10,
  check = FALSE
)

saveRDS(benchmark_results,
        "paper_data_bench-mark_gpu.rds")
