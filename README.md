# Large-Scale Hierarchical Time-Series Reconciliation

[hts.ml.works](http://hts.ml.works)


## Building

1. Install Eigen 3 for OpenMP / LAPACK / BLAS support on Matrix operations
2. Clone this repository with pybind11 submodule
3. `python -m pip install large_scale_hts_reconciliation/` to install the pip package with CMake
4. Test with the following code snippet: 
```
import numpy as np
import lhts
A = np.array([[1,2],[1,1]])
print(lhts.inv(A))
```