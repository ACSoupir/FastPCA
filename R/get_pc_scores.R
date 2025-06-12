#' Calculate the PC scores
#'
#' @param fastpca_out List: output from running FastPCA::FastPCA with elements U, S, and Vh
#'
#' @returns Matrix: contains the scores associated with each of the PCs
#' @export
#'
get_pc_scores = function(fastpca_out){
  #multiply the left singular values by the singular values
  scores = fastpca_out$U %*% diag(fastpca_out$S) #matrix with rows as samples and columns as scores
  #set back the correct column names
  colnames(scores) = colnames(fastpca_out$U)
  return(scores)
}
