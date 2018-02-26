# SourceSelection

Python module to build background galaxy samples for cluster weak lensing analysis.

## Requirements
Python 2.6+, numpy, astropy, corner, scipy, matplotlib.
Also uses functions from the WL.py and useful_el.py modules.

## Install
simply clone all modules included here into your python path.

## Overview

This module contains function that select, displays and analyses the weak lensing signal of background galaxies selected to lie behind galaxy clusters. 
The color-color selection method predominately used here has been developed and describe fully in Medezinski et al. (2010), and rcently further developed and demonstrated in Medezinski et al. (2017). If you find these useful, please cite these papers.

## Examples

### plotting color-color diagram

This exmaple produced a number density map in color-color space
```python
import ColorColor as CC
plotargs = {'color':'0.25','label':'All','s':2}
obj = CC.CCplot(sample,'z','r','i','g',axis=[[-1, 2.5],[-0.5, 3.5]],
                plotpoints=False, levels = [0.2,0.90],**plotargs)
```
