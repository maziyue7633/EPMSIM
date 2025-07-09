# EPMSIM
EPMSIM combines STL decomposition with a ConvBiLSTM network for trend and seasonal reconstruction and a process-based spatiotemporal dynamic tracking interpolation method (PSDTIM) for evolutionary processes, then additively fuses these components to achieve high-accuracy marine spatiotemporal interpolation.

# Setup
All code was developed and tested on NVIDIA GeForce RTX 4060 GPU in the following environment:
- Python 3.10
- torch
- matplotlib
- netCDF4
- scipy
- scikit-learn
- scikit-image
- numpy
- cuda>=12.1
- cudnn>=8.8.1

# Installation

```bash
git clone https://github.com/maziyue7633/EPMSIM.git
cd EPMSIM
```

# Running the Code
```bash
cd epmsim
python main.py
```
# Dependencies
- `matplotlib`: for creating static, animated, and interactive visualizations.
- `netCDF4`: for reading and writing NetCDF files.
- `scipy`: for scientific and numerical computations.
- `scikit-learn`: for machine learning algorithms.
- `scikit-image`: for image processing tasks.
- `numpy`: for numerical operations and array manipulation.


