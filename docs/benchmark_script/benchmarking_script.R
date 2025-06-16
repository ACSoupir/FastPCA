source("docs/benchmark_script/0.1.project_functions.R")
library(pcaone)
library(Matrix)
library(magrittr)
library(irlba)
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
pcaone_alg1_mem = profmem::profmem(pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1"))
saveRDS(pcaone_alg1_res, "docs/benchmark_script/outputs/pcaone_alg1_res.rds")
saveRDS(pcaone_alg1_time, "docs/benchmark_script/outputs/pcaone_alg1_time.rds")
saveRDS(pcaone_alg1_mem, "docs/benchmark_script/outputs/pcaone_alg1_mem.rds")

pcaone_alg2_time = system.time({
  pcaone_alg2_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg2")
})
pcaone_alg2_mem = profmem::profmem(pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg2"))
saveRDS(pcaone_alg2_res, "docs/benchmark_script/outputs/pcaone_alg2_res.rds")
saveRDS(pcaone_alg2_time, "docs/benchmark_script/outputs/pcaone_alg2_time.rds")
saveRDS(pcaone_alg2_mem, "docs/benchmark_script/outputs/pcaone_alg2_mem.rds")

#fastPCA
fastpca_time = system.time({
  fastpca_res = FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations)
})
fastpca_mem = profmem::profmem(expr = FastPCA::FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations))
saveRDS(fastpca_res, "docs/benchmark_script/outputs/fastpca_randomized_cpu_res.rds")
saveRDS(fastpca_time, "docs/benchmark_script/outputs/fastpca_randomized_cpu_time.rds")
saveRDS(fastpca_mem, "docs/benchmark_script/outputs/fastpca_randomized_cpu_mem.rds")

fastpca_exact_time = system.time({
  fastpca_exact_res = FastPCA(scaled_mat, exact = TRUE)
})
fastpca_exact_mem = profmem::profmem(expr = FastPCA(scaled_mat, exact = TRUE))
fastpca_exact_res$U = fastpca_exact_res$U[,1:100] #only keep the first 100
fastpca_exact_res$Vh = fastpca_exact_res$Vh[1:100,]
fastpca_exact_res$S = fastpca_exact_res$S[1:100]
saveRDS(fastpca_exact_res, "docs/benchmark_script/outputs/fastpca_exact_cpu_res.rds")
saveRDS(fastpca_exact_time, "docs/benchmark_script/outputs/fastpca_exact_cpu_time.rds")
saveRDS(fastpca_exact_mem, "docs/benchmark_script/outputs/fastpca_exact_cpu_mem.rds")

#common irlba
irlba_time = system.time({
  irlba_res = irlba(scaled_mat, nv = 100, work = 200) #work = nv + additional subspace
})
irlba_mem = profmem::profmem(irlba(scaled_mat, nv = 100, work = 200))
saveRDS(irlba_res, "docs/benchmark_script/outputs/irlba_res.rds")
saveRDS(irlba_time, "docs/benchmark_script/outputs/irlba_time.rds")
saveRDS(irlba_mem, "docs/benchmark_script/outputs/irlba_mem.rds")
