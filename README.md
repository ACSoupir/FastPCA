# FastPCA

Install with

```
devtools::install_github("ACSoupir/FastPCA")
```

Requires reticulate. Recommend having conda installed for environment isolation.

After installing, need to setup with `FastPCA::setup_py_env()`

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

User - 1.754
System - 1.845
Elapsed - 0.648
