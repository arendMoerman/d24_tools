# Data reduction pipeline for DESHIMA 2.0 data of the 2024 campaign

This Python package contains my data reduction and spectrum stacking for the DESHIMA 2.0 data.

## Installation
Navigate to the directory containing this README, and type:
```
pip install .
```

In case you would like to edit the package to suit your own needs, install it in editable mode:
```
pip install -e .
```

## Reduction of pswsc data
For pswsc data reduction, the flow is as follows:
* despiking
* nodding overshoot removal
* atmosphere subtraction
* grouping and averaging of ABBA cycles

This produces a single spectrum, which can be stacked together with other spectra. 
It can also be rebinned in case this is desired.

See example usage in `reduce_pswsc.py`.

## Reduction of daisy AB data
For daisy with AB chopping data, the procedure is as follows:
* despiking
* atmosphere subtraction
* off-source residual removal

This is currently a work in progress. This repo will be updated when new updates come out.

## Questions, remarks, and contact
Please send to:
A.Moerman (dash) 1 (at) tudelft (dot) nl
