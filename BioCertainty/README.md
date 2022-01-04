# Python Package for certainty in bio-scholarly statements

## How to install the package
Install via `pip`:

```
sudo pip install biocertainty
sudo pip install git+https://github.com/Guindillator/BioCertainty.git
```

 - TODO: add to [PyPI](https://pypi.org/)

## Usage example
```python
Import biocertainty as B

a = PMCID or DOI
b = statement
c  = URL #url where nanopublications would be placed

certainty = B.Certainty(b)

#Either PMCID or DOI
nanopublication = B.Nanopublication(a, b, c)

# 'a' MUST be DOI
micropublication = B.Micropublication (a, b)

## Documentation
* Documentation can be found in the `docs` folder.

## Software citation
Mario Prieto Godoy & Mark Wilkinson. (2019, September 19). Guindillator/BioCertainty: First release of BioCertainty package (Version v1.0.7). Zenodo. http://doi.org/10.5281/zenodo.3571277	

# License
Copyright (c) 2019, Wilkinson Laboratory for Biological Informatics

MIT license
