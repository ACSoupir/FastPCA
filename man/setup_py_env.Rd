% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/setup_py_env.R
\name{setup_py_env}
\alias{setup_py_env}
\title{Set up python environment for FastPCA}
\usage{
setup_py_env(
  method = c("conda", "virtualenv"),
  envname = "FastPCA",
  cuda = FALSE,
  ...
)
}
\arguments{
\item{method}{Character string. The method to use for environment creation.
Can be "virtualenv" (default) or "conda".}

\item{envname}{Character string. The name of the Python environment to create/use.
Defaults to "FastPCA".}

\item{...}{Additional arguments passed to \code{reticulate::virtualenv_create()}
or \code{reticulate::conda_create()}.}

\item{python_version}{Character string. The Python version to use (e.g., "3.9").
Defaults to "3.9". It's recommended to stick to well-supported versions.}
}
\value{
Invisibly returns \code{TRUE} if setup is successful, \code{FALSE} otherwise.
}
\description{
This function ensures that a suitable Python environment (virtualenv or conda)
is available and that 'torch' and 'numpy' are installed within it.
Users should run this once before using functions that rely on PyTorch.
}
\examples{
\dontrun{
  #for conda
  FastPCA::setup_py_env(method = "conda")

  #reticulate virtualenv:
  FastPCA::setup_py_env()
}
}
