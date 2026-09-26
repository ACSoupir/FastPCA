library(magrittr)
library(parallel)
commands = paste0("Rscript --vanilla ", list.files("docs_acs/paper", pattern = "2.0", full.names = TRUE) %>% grep("\\.R", ., value = TRUE))
tmp = mclapply(commands, system, mc.cores = 4, mc.preschedule = FALSE)
#Rscript --vanilla docs_acs/paper/3.00.running_memory.R
