---
# Example from https://joss.readthedocs.io/en/latest/submitting.html
title: 'FastPCA: An R package for fast single value decomposition'
tags:
  - R
  - Python
  - single-cell
  - multiomics
  - reticulate #?
authors:
  - name: Kimberly R. Ward
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Mitchell Hayes
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Steven Eschrich
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Alex C Soupir
    orcid: 0000-0000-0000-0000
    affiliation: "2, 3" # (Multiple affiliations must be quoted)
affiliations:
  - name: Department of Cutaneous Oncology, Moffitt Cancer Center
    index: 1
  - name: Department of Genitourinary Oncology, Moffitt Cancer Center
    index: 2
  - name: Department of Bioinformatics and Biostatistics, Moffitt Cancer Center
    index: 3
citation_author: Ward, Hayes, and Soupir
date: "04 September, 2025"
year: "2025"
bibliography: paper.bib
output: rticles::joss_article
csl: apa.csl
journal: JOSS
---

# Summary

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

``Gala`` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for ``Gala`` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. ``Gala`` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the ``Astropy`` package [@astropy] (``astropy.units`` and
``astropy.coordinates``).

``Gala`` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in ``Gala`` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](https://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Rendered R Figures

Figures can be plotted like so:


``` r
plot(1:10)
```

![](articles_files/figure-latex/unnamed-chunk-1-1.pdf)<!-- --> 


# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

