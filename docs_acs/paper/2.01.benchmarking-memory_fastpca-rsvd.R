#/usr/bin/time -l Rscript --vanilla docs_acs/benchmark_script/testing_bin-time.R
source("docs_acs/benchmark_script/0.1.project_functions.R")
library(FastPCA)
setup_py_env()
start_FastPCA_env()

#read in data
scaled_mat = readRDS("data/prepped_gex.rds")

k = 100
oversampling = 10
power_iterations = 10

tmpf = "docs_acs/paper/2.01.benchmarking-memory_fastpca-rsvd.csv"
try({
  prof <- memprof::with_monitor(
    force({
      FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch")
    }),
    monitor_file = tmpf,
    poll_interval = 0.01,
    overwrite = TRUE
  )
})


val<-bench_time_mem(
  {
    FastPCA(scaled_mat, k = k, p = oversampling, q_iter = power_iterations, backend = "pytorch")
  }
)
saveRDS(val,
        gsub("csv", "rds", tmpf))
