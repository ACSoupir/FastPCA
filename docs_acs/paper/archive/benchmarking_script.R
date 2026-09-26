source("docs_acs/benchmark_script/0.1.project_functions.R")
library(pcaone)
library(Matrix)
library(magrittr)
library(irlba)
#devtools::load_all()
library(FastPCA)
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
#pcaone uses different parameters for power iterations and oversampling
pcaone_alg1_time = system.time({
  pcaone_alg1_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1") #p is power iterations and q is oversampling
})
pcaone_alg1_mem = profmem::profmem(pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1"))
#saveRDS(pcaone_alg1_res, "docs/benchmark_script/outputs/pcaone_alg1_res.rds")
#saveRDS(pcaone_alg1_time, "docs/benchmark_script/outputs/pcaone_alg1_time.rds")
#saveRDS(pcaone_alg1_mem, "docs/benchmark_script/outputs/pcaone_alg1_mem.rds")

pcaone_alg2_time = system.time({
  pcaone_alg2_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg2")
})
pcaone_alg2_mem = profmem::profmem(pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg2"))
#saveRDS(pcaone_alg2_res, "docs/benchmark_script/outputs/pcaone_alg2_res.rds")
#saveRDS(pcaone_alg2_time, "docs/benchmark_script/outputs/pcaone_alg2_time.rds")
#saveRDS(pcaone_alg2_mem, "docs/benchmark_script/outputs/pcaone_alg2_mem.rds")

#common irlba
irlba_time = system.time({
  irlba_res = irlba(scaled_mat, nv = k, work = k + oversampling) #work = nv + additional subspace
})
irlba_mem = profmem::profmem(irlba(scaled_mat, nv = k, work = k + oversampling))
#saveRDS(irlba_res, "docs/benchmark_script/outputs/irlba_res.rds")
#saveRDS(irlba_time, "docs/benchmark_script/outputs/irlba_time.rds")
#saveRDS(irlba_mem, "docs/benchmark_script/outputs/irlba_mem.rds")

#bigstatsr
A = bigstatsr::as_FBM(scaled_mat, type="double")
bigstatsr_partial_time = system.time({
  bigstatsr_partial_res = bigstatsr::big_SVD(A, k = k)
})
#random
bigstatsr_randomSVD_time = system.time({
  bigstatsr_randomSVD_res = bigstatsr::big_randomSVD(A, k = k, ncores = 4)
})
bigstatsr_randomSVD_mem = profmem::profmem(bigstatsr::big_randomSVD(A, k = k, ncores = 4))
#saveRDS(bigstatsr_partial_res, "docs/benchmark_script/outputs/bigstatsr_partial_res.rds")
#saveRDS(bigstatsr_partial_time, "docs/benchmark_script/outputs/bigstatsr_partial_time.rds")
#saveRDS(bigstatsr_partial_mem, "docs/benchmark_script/outputs/bigstatsr_partial_mem.rds")

#fastPCA
fastpca_time_c1 = system.time({
  fastpca_res_c1 = FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch")
})
fastpca_mem_c1 = profmem::profmem(expr = FastPCA::FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch"))
#saveRDS(fastpca_res, "docs/benchmark_script/outputs/fastpca_randomized_cpu_res.rds")
#saveRDS(fastpca_time, "docs/benchmark_script/outputs/fastpca_randomized_cpu_time.rds")
#saveRDS(fastpca_mem, "docs/benchmark_script/outputs/fastpca_randomized_cpu_mem.rds")

fastpca_exact_time_c1 = system.time({
  fastpca_exact_res = FastPCA(scaled_mat, exact = TRUE, backend = "pytorch", cores = 1)
})
fastpca_exact_mem_c1 = profmem::profmem(expr = FastPCA(scaled_mat, exact = TRUE, backend = "pytorch", cores = 1))
fastpca_exact_res$U = fastpca_exact_res$U[,1:100] #only keep the first 100
fastpca_exact_res$Vh = fastpca_exact_res$Vh[1:100,]
fastpca_exact_res$S = fastpca_exact_res$S[1:100]
#saveRDS(fastpca_exact_res, "docs/benchmark_script/outputs/fastpca_exact_cpu_res.rds")
#saveRDS(fastpca_exact_time, "docs/benchmark_script/outputs/fastpca_exact_cpu_time.rds")
#saveRDS(fastpca_exact_mem, "docs/benchmark_script/outputs/fastpca_exact_cpu_mem.rds")

diff_mems_vals = grep("mem", ls(), value = TRUE) %>% setNames(.,.)
diff_mems = lapply(diff_mems_vals[c(4,3,5,6,7,1,2)], get)

diff_times_vals = grep("time", ls(), value = TRUE) %>% setNames(.,.)
diff_times = lapply(diff_times_vals[c(4,3,5,6,7,1,2)], get)

diff_res_vals = grep("res", ls(), value = TRUE) %>% setNames(.,.)
diff_res = lapply(diff_res_vals[c(4,3,5,6,7,1,2)], get)

#all_results
all_results = list(memory = diff_mems,
                   times = diff_times,
                   results = diff_res)
saveRDS.zst(all_results,
            "docs_acs/paper/paper_results.rds", compression_level = 19)

key = data.frame(Method = c("FastPCA (RSVD)", "FastPCA (Exact)", "IRLBA", "PCAone (alg1)", "PCAone (alg2)", "bigstatsr Partial"),
           vars = names(all_results$times) %>% gsub("\\_time", "", .))

#time data
time_table = lapply(all_results$times, function(x) data.frame(t(data.frame(x)))) %>%
  dplyr::bind_rows(.id = "vars") %>%
  dplyr::rename("User Time (s)" = 2, "System Time (s)" = 3, "Elapsed Time (s)" = 4) %>%
  tibble::rownames_to_column("drop") %>%
  dplyr::select(vars, dplyr::contains("Time")) %>%
  dplyr::full_join(key %>%
                     dplyr::mutate(vars = paste0(vars, "_time")),
                   .) %>%
  dplyr::select(-vars)
#memory data
memory_table = lapply(all_results$memory, function(x) data.frame(Memory = utils:::format.object_size(sum(x$bytes, na.rm = TRUE), "auto"))) %>%
  dplyr::bind_rows(.id = "vars")%>%
  dplyr::full_join(key %>%
                     dplyr::mutate(vars = paste0(vars, "_mem")),
                   .) %>%
  dplyr::select(-vars)
#plotting results
singular_table = lapply(all_results$results, function(x){
  if("S" %in% names(x)){
    x$S
  } else {
    x$d
  }
}) %>%
  as.data.frame() %>%
  dplyr::mutate(Dimension = paste0("PC", 1:dplyr::n()),
                .before = 1) %>%
  tidyr::pivot_longer(-Dimension, names_to = "method", values_to = "Eigenvalues")
combos = expand.grid(`m2` = unique(singular_table$method),
                     `m1` = unique(singular_table$method)) %>%
  data.frame() %>%
  dplyr::filter(m2 != m1)

eigenvalues2 = singular_table %>%
  dplyr::rename("m1" = "method", "e1" = "Eigenvalues") %>%
  dplyr::full_join(combos, by = c("m1" = "m1"),
                   relationship = "many-to-many") %>%
  dplyr::full_join(singular_table %>%
                     dplyr::rename("m2" = "method", "e2" = "Eigenvalues"),
                   relationship = "many-to-many",
                   by = dplyr::join_by(Dimension, m2)) %>%
  dplyr::mutate(`Method 1` = key$Method[match(m1, paste0(key$vars, "_res"))],
                `Method 2` = key$Method[match(m2, paste0(key$vars, "_res"))],
                `Method 1` = factor(`Method 1`, levels = unique(key$Method)),
                `Method 2` = factor(`Method 2`, levels = unique(key$Method)))
final_res = list(time = time_table,
                 memory = memory_table,
                 plot_data = eigenvalues2)
saveRDS(final_res,
            "docs_acs/paper/paper_data.rds")


pl = eigenvalues2 %>%
  ggplot() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  geom_point(aes(x = log2(e1), y = log2(e2)), shape = 20, alpha = 0.4, stroke = NA, size = 5) +
  facet_grid(`Method 1` ~ `Method 2`) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  coord_equal() +
  labs(x = "Eigen Values (log2)", y = "Eigen Values (log2)")


png("docs_acs/paper/figure1.png", height = 3500, width = 3500, res = 300)
pl
dev.off()


