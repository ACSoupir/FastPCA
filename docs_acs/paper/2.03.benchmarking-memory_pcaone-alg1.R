#/usr/bin/time -l Rscript --vanilla docs_acs/benchmark_script/testing_bin-time.R
source("docs_acs/benchmark_script/0.1.project_functions.R")
library(pcaone)
#read in data
scaled_mat = readRDS("data/prepped_gex.rds")

k = 100
oversampling = 10
power_iterations = 10

#pcaone_alg1_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1")
tmpf = "docs_acs/paper/2.03.benchmarking-memory_pcaone-alg1.csv"
try({
  prof <- memprof::with_monitor(
    force({
      pcaone_alg1_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1")
    }),
    monitor_file = tmpf,
    poll_interval = 0.01,
    overwrite = TRUE
  )
})


val<-bench_time_mem(
  {
    pcaone_alg1_res =  pcaone(scaled_mat, k = k, q = oversampling, p = power_iterations, method = "alg1")
  }
)
saveRDS(val,
        gsub("csv", "rds", tmpf))

