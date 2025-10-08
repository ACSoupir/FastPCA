## code to prepare `gex` dataset goes here
#https://zenodo.org/doi/10.5281/zenodo.12730226
obj = readRDS("/Volumes/lab_soupir/spatial_transcriptomics/example_data/seurat_object.Rds")
gex = t(apply(as.matrix(obj@assays$Nanostring@counts), 1, as.integer))
saveRDS(gex, "data/gex.rds")
