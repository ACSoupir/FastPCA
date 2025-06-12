#source("0.1.project_functions.R")
library(pcaone)
library(Matrix)
library(magrittr)
FastPCA::setup_py_env()
devtools::load_all()

peaks = readRDS("data/1.0.slide27213peakProccess-ed_peaks.rds")
scaling = readRDS("data/1.0.slide27213peakProccess-ed_scalings.rds")
peak_scaled = sweep(peaks, 2, scaling$scaling_sum, `/`)
peak_norm = sweep(peak_scaled, 2, scaling$scaling_n, `*`)
peak_norm = Matrix(peak_norm, sparse = TRUE)
peak_norm@x = log2(peak_norm@x)
rm(peaks, scaling, peak_scaled)
gc(full=TRUE)
#also transposes now
Rcpp::sourceCpp("docs/benchmark_script/row_scale_sparse.cpp")
scaled_mat <- row_scale_sparse(peak_norm)
rownames(scaled_mat) = colnames(peak_norm)
colnames(scaled_mat) = rownames(peak_norm)

k = 100
oversampling = 10
power_iterations = 10

#pcaone uses different parameters for power iterations and oversampling
pcaone_alg1_time = system.time({
  pcaone_alg1_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1") #p is power iterations and q is oversampling
})
pcaone_alg2_time = system.time({
  pcaone_alg2_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg2")
})
fastpca_time = system.time({
  fastpca_res = FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations)
})
fastpca_exact_time = system.time({
  fastpca_exact_res = FastPCA(scaled_mat, exact = TRUE)
})

pcaone_alg1_mem = profmem::profmem(pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1"))
pcaone_alg2_mem = profmem::profmem(pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg2"))
fastpca_mem = profmem::profmem(expr = FastPCA::FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations))
fastpca_exact_mem = profmem::profmem(expr = FastPCA(scaled_mat, exact = TRUE))

#time
times = data.frame(`PCAone Alg1` = pcaone_alg1_time,
                   `PCAone Alg2` = pcaone_alg2_time,
                   `FastPCA Randomized (mine)` = fastpca_time,
                   `FastPCA Exact (mine)` = fastpca_exact_time,
                   check.names = FALSE)
#memory
mem = data.frame(`PCAone Alg1` = sum(pcaone_alg1_mem$bytes),
                 `PCAone Alg2` = sum(pcaone_alg2_mem$bytes),
                 `FastPCA (mine)` = sum(fastpca_mem$bytes),
                 `FastPCA Randomized (mine)` = sum(fastpca_mem$bytes),
                 `FastPCA Exact (mine)` = sum(fastpca_exact_mem$bytes),
                 check.names = FALSE)
times
mem

pcaone_alg1_time
pcaone_alg2_time
fastpac_time

#check variance explained
full_var = sum(peak_norm^2)

pcaone_alg1_eig_value = pcaone_alg1_res$d
pcaone_alg2_eig_value = pcaone_alg2_res$d
fastpca_eig_value = fastpca_res$S

pcaone_alg1_eig_value
pcaone_alg2_eig_value
fastpca_eig_value

par(mfrow = c(1, 3))
plot(log2(pcaone_alg1_eig_value), log2(pcaone_alg2_eig_value))
abline(0, 1, col = 'red')
plot(log2(pcaone_alg1_eig_value), log2(fastpca_eig_value))
abline(0, 1, col = 'red')
plot(log2(pcaone_alg2_eig_value), log2(fastpca_eig_value))
abline(0, 1, col = 'red')

plot(pcaone_alg1_res$u[,1], pcaone_alg2_res$u[,1])
abline(0, 1, col = 'red')
plot(pcaone_alg1_res$u[,1], fastpca_res$U[,1])
abline(0, 1, col = 'red')
plot(pcaone_alg2_res$u[,1], fastpca_res$U[,1])
abline(0, 1, col = 'red')
