# EPMSIM
EPMSIM combines STL decomposition with a ConvBiLSTM network for trend and seasonal reconstruction and a process-based spatiotemporal dynamic tracking interpolation method (PSDTIM) for evolutionary processes, then additively fuses these components to achieve high-accuracy marine spatiotemporal interpolation.

# Setup
All code was developed and tested on NVIDIA GeForce RTX 4060 Laptop GPU in the following environment:
Python 3.10
torch
matplotlib
netCDF4
scipy
scikit-learn
scikit-image
numpy
cuda>=12.1
cudnn>=8.8.1

# Project Structure
myfunction.py: Utility/helper functions module

data_loader.py: Data loading module

conBiLSTM_model.py: Convolutional Bidirectional LSTM model module

STL_to_NC.py: STL time-series decomposition to NetCDF script

trend_data_loader.py: Trend component data loader

seasonal_data_loader.py: Seasonal component data loader

PSDTIM.py: Process-based spatiotemporal dynamic tracking interpolation module

STL_result_combined.py: Main script combining STL results

test.py: Test script
