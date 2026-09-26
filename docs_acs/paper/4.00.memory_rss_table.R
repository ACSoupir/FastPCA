library(magrittr)
files = list.files("docs_acs/paper", full.names = TRUE, pattern = "csv") %>%
  grep("sensitiv", ., value = TRUE, invert = TRUE)

res = lapply(files, function(f){
  prof = read.table(f, skip = 4, header = TRUE, sep = ',')
  prof = prof %>% dplyr::group_by(time) %>% dplyr::summarise(rss = sum(rss, na.rm = TRUE))

  base <- prof$rss[1]
  peak <- max(prof$rss, na.rm = TRUE)
  delta <- peak - base

  data.frame(
    baseline = base,
    peak = peak,
    peak_increment = delta,
    baseline_pretty = format(structure(base, class = "object_size") , units = "Mb"),
    peak_pretty = format(structure(peak, class = "object_size") , units = "Mb"),
    peak_increment_pretty = format(structure(delta, class = "object_size") , units = "Mb")
  )
}) %>%
  dplyr::bind_rows() %>%
  dplyr::mutate(fun = basename(files),
                name = c("FastPCA (rSVD)", "FastPCA (Exact)",
                         "PCAone (Alg1)", "PCAone (Alg2)",
                         "IRLBA",
                         "bigstatsr (Partial)", "bigstatsr (rSVD)",
                         "BiocSingular (SVD)"))
res

chris_files = list.files("docs_acs/paper", full.names = TRUE, pattern = "rds") %>%
  grep("2.0", ., value = TRUE)
chris_res = lapply(chris_files, function(f){
  d = readRDS(f)
  data.frame(
    chris_delta = format(structure(d['delta'], class = "object_size") , units = "Mb")
  )
}) %>%
  dplyr::bind_rows() %>%
  dplyr::mutate(fun = basename(chris_files) %>%
                  gsub("\\.rds", "", .))

final_memory = dplyr::full_join(
  chris_res,
  res %>% dplyr::mutate(fun = gsub("\\.csv", "", fun))
) %>%
  dplyr::select(name, fun, peak_increment_pretty, chris_delta)
