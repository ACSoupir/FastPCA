#/usr/bin/time -l Rscript --vanilla docs_acs/benchmark_script/testing_bin-time.R
source("docs_acs/benchmark_script/0.1.project_functions.R")
library(irlba)
#read in data
scaled_mat = readRDS("data/prepped_gex.rds")

k = 100
oversampling = 10
power_iterations = 10

#irlba_res = irlba(scaled_mat, nv = k, work = k + oversampling)
tmpf = "docs_acs/paper/2.05.benchmarking-memory_irlba.csv"
try({
  prof <- memprof::with_monitor(
    force({
      irlba_res = irlba(scaled_mat, nv = k, work = k + oversampling)
    }),
    monitor_file = tmpf,
    poll_interval = 0.01,
    overwrite = TRUE
  )
})


val<-bench_time_mem(
  {
    irlba_res = irlba(scaled_mat, nv = k, work = k + oversampling)
  }
)
saveRDS(val,
        gsub("csv", "rds", tmpf))

