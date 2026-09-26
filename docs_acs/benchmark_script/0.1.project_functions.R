#https://gist.github.com/retrography/359e0cc56d2cf1acd161b5645bc801a8
library(parallel)

cmdAvail <- function(cmd) as.logical(nchar(Sys.which(cmd)))

writeRDS <- function(object, con) {
  tryCatch({
    base::saveRDS(
      object,
      file = con
    )
  }, warning = function(w) {
    print(paste("WARNING: ", w))
  }, error = function(e) {
    print(paste("ERROR: ", e))
  }, finally = {
    close(con)
  })
}

loadRDS <- function(con) {
  tryCatch({
    base::readRDS(
      file = con
    )
  }, warning = function(w) {
    print(paste("WARNING: ", w))
  }, error = function(e) {
    print(paste("ERROR: ", e))
  }, finally = {
    close(con)
  })
}

saveRDS.gz <-
  function(object,
           file,
           threads = parallel::detectCores(),
           compression_level = 6) {
    if (cmdAvail("pigz")) {
      writeRDS(
        object,
        pipe(
          paste0("pigz -c -k -p", threads, " -",compression_level," > ", file),
          "wb"
        )
      )
    } else {
      base::saveRDS(object, file = file, compress = "gzip")
    }
  }

readRDS.gz <-
  function(file,
           threads = parallel::detectCores()) {
    if (cmdAvail("pigz")) {
      object <-
        loadRDS(
          pipe(
            paste0("pigz -dc -p", threads, " ", file)
          )
        )
    } else {
      object <-
        base::readRDS(file)
    }
    return(object)
  }

#using zstd instead
saveRDS.zst <-
  function(object,
           file,
           threads = parallel::detectCores(),
           compression_level = 17) {
    if (cmdAvail("zstd")) {
      writeRDS(
        object,
        pipe(
          paste0("zstd -q --threads ", threads, " -", compression_level," -c > ", shQuote(file)),
          "wb"
        )
      )
    } else {
      base::saveRDS(object, file = file, compress = "gzip")
    }
  }

readRDS.zst <-
  function(file) {
    if (cmdAvail("zstd")) {
      object <-
        loadRDS(
          pipe(
            paste0("zstd -dc ", file)
          )
        )
    } else {
      object <-
        base::readRDS(file)
    }
    return(object)
  }

#custom
get_slurm_out = function (slr_job, outtype = "raw", wait = TRUE, ncores = NULL)
{
  if (!(inherits(slr_job, "slurm_job"))) {
    stop("slr_job must be a slurm_job")
  }
  outtypes <- c("table", "raw")
  if (!(outtype %in% outtypes)) {
    stop(paste("outtype should be one of:", paste(outtypes,
                                                  collapse = ", ")))
  }
  if (!(is.null(ncores) || (is.numeric(ncores) && length(ncores) ==
                            1))) {
    stop("ncores must be an integer number of cores")
  }
  if (wait) {
    rslurm:::wait_for_job(slr_job)
  }
  res_files <- paste0("results_", 0:(slr_job$nodes - 1), ".RDS")
  tmpdir <- paste0("_rslurm_", slr_job$jobname)
  missing_files <- setdiff(res_files, dir(path = tmpdir))
  if (length(missing_files) > 0) {
    missing_list <- paste(missing_files, collapse = ", ")
    warning(paste("The following files are missing:", missing_list))
  }
  res_files <- file.path(tmpdir, setdiff(res_files, missing_files))
  if (length(res_files) == 0)
    return(NA)
  if (is.null(ncores)) {
    slurm_out <- lapply(res_files, readRDS)
  }
  else {
    slurm_out <- mclapply(res_files, readRDS, mc.cores = ncores)
  }
  slurm_out <- do.call(c, slurm_out)
  if (outtype == "table") {
    slurm_out <- as.data.frame(do.call(rbind, slurm_out))
  }
  slurm_out
}

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
      mem<-profmem::profmem(time<-system.time(x,gcFirst = FALSE))
      rettime<-c(time)
      names(rettime)<-names(time)
      retval<-c(time,"R_gc_alloc" = sum(mem$bytes,na.rm=T))
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
