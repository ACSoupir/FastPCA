#/usr/bin/time -l Rscript --vanilla docs_acs/benchmark_script/testing_bin-time.R
source("docs_acs/benchmark_script/0.1.project_functions.R")
library(BiocSingular)
#read in data
scaled_mat = readRDS("data/prepped_gex.rds")

k = 100
oversampling = 10
power_iterations = 10

#fastpca_exact_res = FastPCA(scaled_mat, exact = TRUE, backend = "pytorch", cores = 1)
tmpf = "docs_acs/paper/2.08.benchmarking-memory_biocsingular.csv"
try({
  prof <- memprof::with_monitor(
    force({
      biosingular = runSVD(scaled_mat, k = k,
                           BPPARAM = BiocParallel::MulticoreParam(workers = 4),
                           BSPARAM = RandomParam(),
                           p = oversampling,
                           q = power_iterations)
    }),
    monitor_file = tmpf,
    poll_interval = 0.01,
    overwrite = TRUE
  )
})

bench_time_mem<-function(x, step_of_monitor = 0.001) {
  gc(reset=FALSE,verbose=FALSE)  ## force a gc here
  pidfile <- tempfile() # temp file to story Perl's PID
  outfile <- tempfile() # stdout redirection for Perl
  pid <- Sys.getpid()   # get R's PID
  ret<-NULL
  system2("./docs_acs/paper/monitor_memory.pl",
          c(pid,step_of_monitor,pidfile),wait=FALSE,stdout=outfile)
  Sys.sleep(0.2)  # Wait for PID file to be written
  monitor_pid <- readLines(pidfile)[1] # Get Perl's PID
  tryCatch (
    expr = {
      time<-system.time(x,gcFirst = FALSE)
      rettime<-c(time)
      names(rettime)<-names(time)
      retval<-c(time,"R_gc_alloc" = NA_real_)
    }, # execute R expression, get timing and allocations
    finally = {
      system2("kill",c("-TERM", monitor_pid)) # kill the ? orphan
      Sys.sleep(0.2) # Wait for Perl to finish logging
      memstats<-read.csv(outfile,sep="\t",
                         header=FALSE) # get memory statistics
      unlink(c(pidfile,outfile)) #cleanup files
      retval<-c(retval ,
                "delta"= memstats[1,1]*1024,
                "initial"= memstats[1,2]*1024
      )
    }
  )

  return(retval)
}


val<-bench_time_mem(
  {
    biosingular = runSVD(scaled_mat, k = k,
                         BPPARAM = BiocParallel::MulticoreParam(workers = 4),
                         BSPARAM = RandomParam(),
                         p = oversampling,
                         q = power_iterations)
  }
)
saveRDS(val,
        gsub("csv", "rds", tmpf))

