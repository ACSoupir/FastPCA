#/usr/bin/time -l Rscript --vanilla docs_acs/benchmark_script/testing_bin-time.R
source("docs_acs/benchmark_script/0.1.project_functions.R")
library(bigstatsr)
#read in data
A = big_attach("data/prepped_gex_fbm.rds")

k = 100
oversampling = 10
power_iterations = 10

#bigstatsr_randomSVD_res = bigstatsr::big_randomSVD(A, k = k, ncores = 4)
tmpf = "docs_acs/paper/2.07.benchmarking-memory_bigstatsr-bigrandomsvd.csv"
try({
  prof <- memprof::with_monitor(
    force({
      bigstatsr_randomSVD_res = bigstatsr::big_randomSVD(A, k = k, ncores = 4)
    }),
    monitor_file = tmpf,
    poll_interval = 0.01,
    overwrite = TRUE
  )
})

val<-bench_time_mem(
  {
    bigstatsr_randomSVD_res = bigstatsr::big_randomSVD(A, k = k, ncores = 4)
  }
)
saveRDS(val,
        gsub("csv", "rds", tmpf))
