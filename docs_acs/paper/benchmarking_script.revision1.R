source("docs_acs/benchmark_script/0.1.project_functions.R")
library(pcaone)
library(Matrix)
library(magrittr)
library(irlba)
#devtools::load_all()
library(FastPCA)
library(BiocSingular)
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
pcaone_alg1_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1") #p is power iterations and q is oversampling
saveRDS.zst(pcaone_alg1_res, "docs_acs/benchmark_script/outputs/pcaone_alg1_res.rds", compression_level = 22)

pcaone_alg2_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg2")
#saveRDS.zst(pcaone_alg2_res, "docs_acs/benchmark_script/outputs/pcaone_alg2_res.rds", compression_level = 22)

#common irlba
irlba_res = irlba(scaled_mat, nv = k, work = k + oversampling)
#saveRDS.zst(irlba_res, "docs_acs/benchmark_script/outputs/irlba_res.rds", compression_level = 22)

#bigstatsr
A = bigstatsr::as_FBM(scaled_mat, type="double")
bigstatsr_partial_res = bigstatsr::big_SVD(A, k = k)
#random
bigstatsr_randomSVD_res = bigstatsr::big_randomSVD(A, k = k, ncores = 4)
#saveRDS.zst(bigstatsr_partial_res, "docs_acs/benchmark_script/outputs/bigstatsr_partial_res.rds", compression_level = 22)

#fastPCA
fastpca_res_c1 = FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch")
#saveRDS.zst(fastpca_res_c1, "docs_acs/benchmark_script/outputs/fastpca_randomized_cpu_res.rds", compression_level = 22)

fastpca_exact_res = FastPCA(scaled_mat, exact = TRUE, backend = "pytorch", cores = 1)
# fastpca_exact_res$U = fastpca_exact_res$U[,1:100] #only keep the first 100
# fastpca_exact_res$Vh = fastpca_exact_res$Vh[1:100,]
# fastpca_exact_res$S = fastpca_exact_res$S[1:100]
#saveRDS.zst(fastpca_exact_res, "docs_acs/benchmark_script/outputs/fastpca_exact_cpu_res.rds", compression_level = 22)


biosingular_res = runSVD(scaled_mat, k = k,
                     BPPARAM = BiocParallel::MulticoreParam(workers = 16),
                     BSPARAM = RandomParam(),
                     p = oversampling,
                     q = power_iterations)
#saveRDS.zst(biosingular_res, "docs_acs/benchmark_script/outputs/biosingular_res.rds", compression_level = 22)

diff_res_vals = grep("res", ls(), value = TRUE) %>% setNames(.,.)
diff_res = lapply(diff_res_vals[c(5,4,6,7,8,2,1,3)], get)

#reconstruction errors
#reference
Uref = diff_res$fastpca_exact_res$U
Sref = diff_res$fastpca_exact_res$S
Vref = diff_res$fastpca_exact_res$Vh %>% t()
XhatR = Uref %*% diag(Sref) %*% t(Vref)
den = sum((scaled_mat - XhatR)^2)

#because we are doing trucated
Uref2 = diff_res$fastpca_exact_res$U[,1:100]
Sref2 = diff_res$fastpca_exact_res$S[1:100]
Vref2 = diff_res$fastpca_exact_res$Vh[1:100,] %>% t()
XhatR2 = Uref2 %*% diag(Sref2) %*% t(Vref2)
den2 = sum((scaled_mat - XhatR2)^2)

errors = lapply(diff_res, function(x){
  if("S" %in% names(x)){
    recon = x$U[,1:100] %*% diag(x$S[1:100]) %*% x$Vh[1:100,]
  } else {
    recon = x$u[,1:100] %*% diag(x$d[1:100]) %*% t(x$v)[1:100,]
  }
  num = sum((scaled_mat - recon)^2)
  return(data.frame(`Full Error` = num / den,
                    `Truncated Error` = num / den2,
                    check.names = FALSE))
}) %>%
  dplyr::bind_rows(.id = "vars")

errors2 = errors %>%
  dplyr::mutate(Algorithm =  c("FastPCA (rSVD)", "FastPCA (Exact)", "IRLBA", "PCAone (alg1)", "PCAone (alg2)",
                               "bigstatsr (rSVD)", "bigstatsr (Partial)", "BiocSingular"))
errors2


#Frobenius norm/absolute error
norm(scaled_mat - XhatR, type = "F")
#relative error
norm(scaled_mat - XhatR, type = "F")/sqrt(sum(scaled_mat^2))


##CCC for correlation - Brooke
#DescTools::CCC
svals = lapply(diff_res, function(x){
  if("S" %in% names(x)){
    x$S[1:100]
  } else {
    x$d[1:100]
  }
})

ref_svd_method = grep("exact", errors2$vars, value = TRUE)
test_svd_methods = grep("exact", errors2$vars, value = TRUE, invert = TRUE)
ccc_singular_value_res = lapply(test_svd_methods, function(m){
  DescTools::CCC(svals[[ref_svd_method]],
                 svals[[m]])
})

PCs = lapply(diff_res, function(x){
  if("S" %in% names(x)){
    res = x$U[,1:100] %*% diag(x$S[1:100])
  } else {
    res = x$u[,1:100] %*% diag(x$d[1:100])
  }
  res[,1:2]
})

#see how the PCs agree.
ccc_pc1_res = lapply(test_svd_methods, function(m){
  if(cor(PCs[[ref_svd_method]][,1], PCs[[m]][,1]) < 0){
    DescTools::CCC(PCs[[ref_svd_method]][,1],
                   PCs[[m]][,1] * -1)
  } else {
    DescTools::CCC(PCs[[ref_svd_method]][,1],
                   PCs[[m]][,1])
  }
})
ccc_pc2_res = lapply(test_svd_methods, function(m){
  if(cor(PCs[[ref_svd_method]][,2], PCs[[m]][,2]) < 0){
    DescTools::CCC(PCs[[ref_svd_method]][,2],
                   PCs[[m]][,2] * -1)
  } else {
    DescTools::CCC(PCs[[ref_svd_method]][,2],
                   PCs[[m]][,2])
  }
})

names(ccc_singular_value_res) = names(ccc_pc1_res) = names(ccc_pc2_res) = test_svd_methods
ccc_res = lapply(list(values = ccc_singular_value_res,
            PC1 = ccc_pc1_res,
            PC2 = ccc_pc2_res),
       function(x){
         lapply(x, function(y) y$rho.c %>% dplyr::select(est)) %>%
           dplyr::bind_rows(.id = "method")
       })
ccc_res = lapply(1:3, function(x){
  colnames(ccc_res[[x]])[2] = c("Singular Values", "Principal Component 1", "Principal Component 2")[x]
  ccc_res[[x]]
}) %>%
  Reduce(dplyr::full_join, .) %>%
  dplyr::inner_join(errors2 %>%
                      dplyr::select(vars, Algorithm) %>%
                      dplyr::rename("method" = vars), .)
ccc_res
