# FastPCA

Install with

```         
devtools::install_github("ACSoupir/FastPCA")
```

Requires reticulate. Recommend having conda installed for environment isolation.

After installing, need to setup with `FastPCA::setup_py_env()`

## Benchmarking against PCAone

Using a matrix that contains 98,647 pixels with 2,925 MALDI peaks, I have run the [`PCAone`](https://cran.r-project.org/web/packages/pcaone/index.html) package with both of their algorithms. For each of the methods, I calculated 100 dimensions from the data using 10 oversampling dimensions as well as 10 power iterations. The speed difference was:

|   | User Time (s) | System Time (s) | **Elapsed Time (s)** |
|----|----|----|----|
| PCAone Alg1 | 44.805 | 0.574 | **45.556** |
| PCAone Alg2 | 48.518 | 0.743 | **49.446** |
| FastPCA Randomized | 20.881 | 5.612 | **6.272** |
| FastPCA Exact | 79.694 | 11.564 | **35.819** |
| FastPCA Randomized (cuda) | 0.799 | 0.638 | **0.939** |

Memory does appear to be greater when using FastPCA over PCAone (profiled with [`profmem`](https://cran.r-project.org/web/packages/profmem/)):

|   | PCAone Alg1 | PCAone Alg2 | FastPCA Randomized (mine) | FastPCA Exact (mine) |
|----|----|----|----|----|
| Memory (MB) | 81.26 | 81.26 | 171.05 | 4753.69 |

|                           | Memory (MB) |
|---------------------------|-------------|
| PCAone Alg1               | 81.26       |
| PCAone Alg2               | 81.26       |
| FastPCA Randomized        | 171.05      |
| FastPCA Exact             | 4753.69     |
| FastPCA Randomized (cuda) | 171.06      |

## Example

```         
library(FastPCA)

setup_py_env(method = "conda", envname = "FastPCA", cuda = FALSE)

start_dat = readRDS("smalley_maldi_clustering_for_alex_2025-06-06/27213_all_regions-nonorm_norm_filtered.rds")
dim(start_dat)
#2343 x 98647

processed_dat = FastPCA::prep_matrix(as.matrix(start_dat),
                                     log2 = TRUE, 
                                     transpose = TRUE,
                                     scale = TRUE)
dim(processed_dat)
#98647 x 2343

out_svd = FastPCA(processed_dat, 
                  k = 50,
                  p = 10,
                  q_iter = 2)
system.time({
  out_svd = FastPCA(processed_dat, 
                    k = 50,
                    p = 10,
                    q_iter = 2)
})
```

Execution times:

-   User - 1.754
-   System - 1.845
-   Elapsed - 0.648

## Outputs

Outputs are singular values. To convert to scores in R, multiply the left singular values by the

```         
torch_pc_scores = get_pc_scores(out_svd)
```
