#/usr/bin/time -l Rscript --vanilla docs_acs/benchmark_script/testing_bin-time.R
source("docs_acs/benchmark_script/0.1.project_functions.R")
library(FastPCA)
setup_py_env()
start_FastPCA_env()

#because needs to compile
run_once_trigger = FastPCA(matrix(rnorm(n = 1000,
                                        mean = 0,
                                        sd = 1),
                                  ncol = 10),
                           k = 5, p = 2, q_iter = 2, backend = "pytorch")
#read in data
scaled_mat = readRDS("data/prepped_gex.rds")


k = 100
oversampling = 10
power_iterations = 10

tmpf = "docs_acs/paper/2.01.benchmarking-memory_fastpca-rsvd_sensitivity1.csv"
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

prof = read.table(tmpf, skip = 4, header = TRUE, sep = ',')
base <- prof$rss[1]
peak <- max(prof$rss, na.rm = TRUE)
delta <- peak - base

list(
  baseline = base,
  peak = peak,
  peak_increment = delta,
  baseline_pretty = format(structure(base, class = "object_size") , units = "Mb"),
  peak_pretty = format(structure(peak, class = "object_size") , units = "Mb"),
  peak_increment_pretty = format(structure(delta, class = "object_size") , units = "Mb")
)

# "2427.9 Mb", "2431.7 Mb", "2431.4 Mb", "2426.6 Mb"
