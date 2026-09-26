import umap
import numpy as np

def umap_learn_py(matrix: np.ndarray, 
                  n_neighbors: int = 15, 
                  n_components: int = 2, 
                  metric: str = 'euclidean', 
                  min_dist: float = 0.1,
                  cores: int = 4,
                  seed: int = 4,
                  densemap: int = 0,
                  dens_lambda: float = 2.0,
                  verbose: int = 0):
  """
  Calculates the UMAP embedding .

  Args:
      matrix: Input matrix from R with shape (samples, features).
      n_neighbors: The size of local neighborhood (in terms of number of neighboring
                   sample points) used for manifold approximation.
      n_components: The dimension of the space to embed into.
      metric: The metric to use to compute distances in high dimensional space.
      min_dist: The effective minimum distance between embedded points.
      cores: The number or CPU cores to use

  Returns:
      A NumPy array of shape (samples, n_components) containing the UMAP embedding.
  """
  #convert densemap back
  if densemap == 1:
    densemap = True
  else:
    densemap = False
  if verbose == 1:
    verbose = True
  else:
    verbose = False
  
  #umap model
  if seed == -1 or cores > 1:
    print("Parallel calculation - Seed unavailable")
    umap_model = umap.UMAP(
      n_neighbors=n_neighbors,
      n_components=n_components,
      min_dist=min_dist,
      metric=metric,
      n_jobs=cores,
      densmap = densemap,
      dens_lambda = dens_lambda,
      verbose = verbose
  )
  else: umap_model = umap.UMAP(
      n_neighbors=n_neighbors,
      n_components=n_components,
      min_dist=min_dist,
      metric=metric,
      n_jobs=cores,
      random_state = seed,
      densmap = densemap,
      dens_lambda = dens_lambda,
      verbose = verbose
  )
  
  
  embedding = umap_model.fit_transform(matrix)
  
  return embedding
