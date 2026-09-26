source("docs_acs/benchmark_script/0.1.project_functions.R")
library(pcaone)
library(Matrix)
library(magrittr)
library(irlba)
library(ggplot2)
library(BiocSingular)
#devtools::load_all()
library(FastPCA)
library(bench)
setup_py_env()
start_FastPCA_env()

#data
gex = readRDS("data/gex.rds")

gc(full=TRUE)
#also transposes now
#convert back to numeric becuase currently is int
gex_vec = as.numeric(gex)
gex2 = matrix(gex_vec, nrow = nrow(gex))
scaled_mat = prep_matrix(gex2+1, transpose = TRUE, backend = "pytorch") #add 1 because pesky 0s

k = 100
oversampling = 10
power_iterations = 10

set.seed(333)

# Bench::Mark -------------------------------------------------------------
A = bigstatsr::as_FBM(scaled_mat, type="double")
benchmark_results = bench::mark(
  pcaone_alg1_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1"), #p is power iterations and q is oversampling
  pcaone_alg2_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg2"),
  irlba_res = irlba(scaled_mat, nv = k, work = k + oversampling), #work = nv + additional subspace
  bigstatsr_partial_res = bigstatsr::big_SVD(A, k = k),
  bigstatsr_randomSVD_res = bigstatsr::big_randomSVD(A, k = k, ncores = 4),
  fastpca_res_c1 = FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch"),
  fastpca_exact_res = FastPCA(scaled_mat, exact = TRUE, backend = "pytorch", cores = 1),
  biosingular = runSVD(scaled_mat, k = k,
                       BPPARAM = BiocParallel::MulticoreParam(workers = 4),
                       BSPARAM = RandomParam(),
                       p = oversampling,
                       q = power_iterations),
  min_iterations = 10,
  check = FALSE
)

out = dplyr::bind_rows(benchmark_results, benchmark_resuls2)

# saveRDS(out,
#         "docs_acs/paper/paper_data_bench-mark.rds")
out = readRDS("docs_acs/paper/paper_data_bench-mark.rds")
gpu_out = readRDS("docs_acs/paper/paper_data_bench-mark_gpu.rds")
out = dplyr::bind_rows(out, gpu_out)

library(bench)
pl = out %>%
  tidyr::unnest(c(time, gc)) %>%
  dplyr::mutate(expression = dplyr::case_when(expression == "pcaone_alg1_res" ~ "PCAone (Alg1)",
                                              expression == "pcaone_alg2_res" ~ "PCAone (Alg2)",
                                              expression == "irlba_res" ~ "IRLBA",
                                              expression == "bigstatsr_partial_res" ~ "bigstatsr (Partial)",
                                              expression == "bigstatsr_randomSVD_res" ~ "bigstatsr (rSVD)",
                                              expression == "fastpca_res_c1" ~ "FastPCA (rSVD)",
                                              expression == "fastpca_exact_res" ~ "FastPCA (Exact)",
                                              expression == "biosingular" ~ "BiocSingular (SVD)",
                                              expression == "fastpca_res_gpu" ~ "FastPCA GPU (rSVD)"),
                expression = factor(expression, levels = c("FastPCA (rSVD)","FastPCA (Exact)","IRLBA","PCAone (Alg1)","PCAone (Alg2)",
                                                           "bigstatsr (rSVD)","bigstatsr (Partial)","BiocSingular (SVD)", "FastPCA GPU (rSVD)"))) %>%
  #dplyr::arrange(expression) %>%
  ggplot() +
  geom_boxplot(aes(x = expression, y = time, color = expression)) +
  scale_color_viridis_d() +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Singular Value Decomposition Implementation",
       y = "Time") +
  guides(color = "none")
pl

png("docs_acs/paper/figure1.png", height = 1600, width = 2000, res = 300)
pl
dev.off()

out %>%
  tidyr::unnest(c(time, gc)) %>%
  dplyr::select(expression, mem_alloc) %>%
  dplyr::distinct() %>%
  dplyr::mutate(expression = dplyr::case_when(expression == "pcaone_alg1_res" ~ "PCAone (Alg1)",
                                              expression == "pcaone_alg2_res" ~ "PCAone (Alg2)",
                                              expression == "irlba_res" ~ "IRLBA",
                                              expression == "bigstatsr_partial_res" ~ "bigstatsr (Partial)",
                                              expression == "bigstatsr_randomSVD_res" ~ "bigstatsr (rSVD)",
                                              expression == "fastpca_res_c1" ~ "FastPCA (rSVD)",
                                              expression == "fastpca_exact_res" ~ "FastPCA (Exact)",
                                              expression == "biosingular" ~ "BiocSingular (SVD)",
                                              expression == "fastpca_res_gpu" ~ "FastPCA GPU (rSVD)"),
                expression = factor(expression, levels = c("FastPCA (rSVD)","FastPCA (Exact)","IRLBA","PCAone (Alg1)","PCAone (Alg2)",
                                                           "bigstatsr (rSVD)","bigstatsr (Partial)","BiocSingular (SVD)", "FastPCA GPU (rSVD)"))) %>%
  dplyr::arrange(expression)

out %>%
  tidyr::unnest(c(time, gc)) %>%
  dplyr::group_by(expression) %>%
  dplyr::summarise(min = min(time),
                   median = median(time),
                   mean = mean(time),
                   max = max(time)) %>%
  dplyr::mutate(expression = dplyr::case_when(expression == "pcaone_alg1_res" ~ "PCAone (Alg1)",
                                              expression == "pcaone_alg2_res" ~ "PCAone (Alg2)",
                                              expression == "irlba_res" ~ "IRLBA",
                                              expression == "bigstatsr_partial_res" ~ "bigstatsr (Partial)",
                                              expression == "bigstatsr_randomSVD_res" ~ "bigstatsr (rSVD)",
                                              expression == "fastpca_res_c1" ~ "FastPCA (rSVD)",
                                              expression == "fastpca_exact_res" ~ "FastPCA (Exact)",
                                              expression == "biosingular" ~ "BiocSingular (SVD)",
                                              expression == "fastpca_res_gpu" ~ "FastPCA GPU (rSVD)"),
                expression = factor(expression, levels = c("FastPCA (rSVD)","FastPCA (Exact)","IRLBA","PCAone (Alg1)","PCAone (Alg2)",
                                                           "bigstatsr (rSVD)","bigstatsr (Partial)","BiocSingular (SVD)", "FastPCA GPU (rSVD)"))) %>%
  dplyr::arrange(expression) %>%
  dplyr::mutate(dplyr::across(min:max, ~as.numeric(.x) %>% round(2))) %>%
  data.frame()
