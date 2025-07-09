import torch
import numpy as np
import matplotlib.pyplot as plt
from conBiLSTM_model import create_model
import os
from datetime import datetime
import netCDF4 as nc
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import netCDF4 as nc

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(parent_dir)
import myfunction as MY

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings

import importlib
from skimage.metrics import structural_similarity as ssim

from PSDTIM import PSDTIM


def load_roi_time_data(nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time, variable):
    """Load data for a specified ROI and time range from a NetCDF file"""
    ds = nc.Dataset(nc_file_path, 'r')
    
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    
    lat_min, lat_max = roi_lat_range
    lon_min, lon_max = roi_lon_range
    lat_min_idx = np.searchsorted(lats, lat_min)
    lat_max_idx = np.searchsorted(lats, lat_max)
    lon_min_idx = np.searchsorted(lons, lon_min)
    lon_max_idx = np.searchsorted(lons, lon_max)
    
    num_times = len(ds.dimensions['time'])
    if start_time < 0 or end_time >= num_times:
        raise IndexError("start_time or end_time out of range of time steps.")
    
    sst_var = ds.variables[variable]
    sst_roi_time = sst_var[start_time:end_time + 1, 
                          lat_min_idx:lat_max_idx, 
                          lon_min_idx:lon_max_idx].astype(np.float32)
    
    lats_roi = lats[lat_min_idx:lat_max_idx]
    lons_roi = lons[lon_min_idx:lon_max_idx]
    
    ds.close()
    return sst_roi_time, lats_roi, lons_roi



class TrendInterpolator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Trend model device: {self.device}")
        
        self.model = create_model(input_channels=1, hidden_channels=8, kernel_size=3)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def interpolate(self, sst_data):
        mean = np.mean(sst_data)
        std = np.std(sst_data)
        sst_normalized = (sst_data - mean) / std
        x = torch.FloatTensor(sst_normalized)[None, :, :, :, None].to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
        
        predicted = output.cpu().numpy()
        predicted = predicted.squeeze() * std + mean
        return predicted

class SeasonalInterpolator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Seasonal model device: {self.device}")
        
        self.model = create_model(input_channels=3, hidden_channels=8, kernel_size=3)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def interpolate(self, sst_data, time_idx, period_length):
        mean = np.mean(sst_data)
        std = np.std(sst_data)
        sst_normalized = (sst_data - mean) / std
        
        phase = 2 * np.pi * time_idx / period_length
        sin_feat = np.sin(phase)
        cos_feat = np.cos(phase)
        
        time_steps, height, width = sst_normalized.shape
        x = np.zeros((1, time_steps, height, width, 3))
        x[0, :, :, :, 0] = sst_normalized
        x[0, :, :, :, 1] = sin_feat
        x[0, :, :, :, 2] = cos_feat
        
        x = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
        
        predicted = output.cpu().numpy()
        predicted = predicted.squeeze() * std + mean
        return predicted



def evaluate_interpolation(predicted, true_value):
    """Evaluate interpolation results"""
    predicted = predicted.squeeze()
    true_value = true_value.squeeze()
    
    mse = np.mean((predicted - true_value) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted - true_value))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
    }

def get_true_sst_data(nc_file_path, target_time, lat_range, lon_range):
    """Get true SST data"""
    with Dataset(nc_file_path, 'r') as nc:
        lats = nc.variables['lat'][:]
        lons = nc.variables['lon'][:]
        
        lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
        lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
        
        sst = nc.variables['analysed_sst'][int(target_time), lat_mask, lon_mask]
        
        # Convert to Celsius
        sst = sst - 273.15
        return sst

def calculate_metrics(true_sst, predicted_sst):
    """Calculate evaluation metrics"""
    # Calculate base metrics
    rmse = np.sqrt(mean_squared_error(true_sst.flatten(), predicted_sst.flatten()))
    mae = mean_absolute_error(true_sst.flatten(), predicted_sst.flatten())
    
    # Calculate SSIM
    data_range = true_sst.max() - true_sst.min()
    ssim_value = ssim(true_sst, predicted_sst, data_range=data_range)
    
    # Calculate R²
    reg = LinearRegression().fit(true_sst.flatten().reshape(-1, 1), predicted_sst.flatten().reshape(-1, 1))
    r2 = reg.score(true_sst.flatten().reshape(-1, 1), predicted_sst.flatten().reshape(-1, 1))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'SSIM': ssim_value,
        'R2': r2
    }

def plot_results(originals, predicteds, combined, true_sst, lats, lons, metrics, save_dir):
    """Plot and save results, including comparison with true values"""
    plt.figure(figsize=(20, 18))
    
    # Compute min/max for each component
    components = ['trend', 'seasonal', 'residual']
    vmins = []
    vmaxs = []
    
    for i, (orig, pred) in enumerate(zip(originals, predicteds)):
        vmin = min(np.min(orig), np.min(pred))
        vmax = max(np.max(orig), np.max(pred))
        vmins.append(vmin)
        vmaxs.append(vmax)
    
    # Plot original data and interpolated results
    for i, (component, orig, pred) in enumerate(zip(components, originals, predicteds)):
        # Original data
        plt.subplot(5, 2, i*2+1)
        plt.pcolormesh(lons, lats, orig, shading='auto', cmap='RdBu_r',
                      vmin=vmins[i], vmax=vmaxs[i])
        plt.colorbar(label='Temperature (°C)')
        plt.title(f'Original {component.capitalize()}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Interpolated results
        plt.subplot(5, 2, i*2+2)
        plt.pcolormesh(lons, lats, pred, shading='auto', cmap='RdBu_r',
                      vmin=vmins[i], vmax=vmaxs[i])
        plt.colorbar(label='Temperature (°C)')
        plt.title(f'Interpolated {component.capitalize()}\nRMSE: {metrics[f"{component}_RMSE"]:.4f}°C')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    
    # Plot combined result and true comparison
    vmin = min(np.min(combined), np.min(true_sst))
    vmax = max(np.max(combined), np.max(true_sst))
    
    # Combined result
    plt.subplot(5, 2, 9)
    plt.pcolormesh(lons, lats, combined, shading='auto', cmap='RdBu_r',
                  vmin=vmin, vmax=vmax)
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Combined Result\nRMSE: {metrics["combined_vs_true_RMSE"]:.4f}°C\n' + 
             f'SSIM: {metrics["SSIM"]:.4f}\nR²: {metrics["R2"]:.4f}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # True values
    plt.subplot(5, 2, 10)
    plt.pcolormesh(lons, lats, true_sst, shading='auto', cmap='RdBu_r',
                  vmin=vmin, vmax=vmax)
    plt.colorbar(label='Temperature (°C)')
    plt.title('True SST')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.figure(figsize=(20, 15))
    
    # Compute min/max for each component
    components = ['trend', 'seasonal', 'residual']
    vmins = []
    vmaxs = []
    
    for i, (orig, pred) in enumerate(zip(originals, predicteds)):
        vmin = min(np.min(orig), np.min(pred))
        vmax = max(np.max(orig), np.max(pred))
        vmins.append(vmin)
        vmaxs.append(vmax)
    
    # Plot original data and interpolated results
    for i, (component, orig, pred) in enumerate(zip(components, originals, predicteds)):
        # Original data
        plt.subplot(4, 2, i*2+1)
        plt.pcolormesh(lons, lats, orig, shading='auto', cmap='RdBu_r',
                      vmin=vmins[i], vmax=vmaxs[i])
        plt.colorbar(label='Temperature (°C)')
        plt.title(f'Original {component.capitalize()}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Interpolated results
        plt.subplot(4, 2, i*2+2)
        plt.pcolormesh(lons, lats, pred, shading='auto', cmap='RdBu_r',
                      vmin=vmins[i], vmax=vmaxs[i])
        plt.colorbar(label='Temperature (°C)')
        plt.title(f'Interpolated {component.capitalize()}\nRMSE: {metrics[f"{component}_RMSE"]:.4f}°C')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    
    # Plot combined result
    plt.subplot(4, 1, 4)
    plt.pcolormesh(lons, lats, combined, shading='auto', cmap='RdBu_r')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Combined Result (Trend + Seasonal + Residual)\nRMSE: {metrics["combined_RMSE"]:.4f}°C')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'full_interpolation_results_{timestamp}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return save_path

def plot_scatter_comparison(true_sst, predicted_sst):
    """Plot scatter comparison of true vs predicted SST"""
    plt.figure(figsize=(10, 8))
    plt.scatter(true_sst.flatten(), predicted_sst.flatten(), alpha=0.5)
    plt.plot([true_sst.min(), true_sst.max()], [true_sst.min(), true_sst.max()], 'r--')
    plt.xlabel('True SST (°C)')
    plt.ylabel('Predicted SST (°C)')
    plt.title('True vs Predicted SST')
    plt.grid(True)
    return plt.gcf()

def main():
    # Parameter settings
    trend_model_path = 'trend_checkpoints_test/best_model2.pth'
    seasonal_model_path = 'seasonal_checkpoints_test/best_model2.pth'
  
    residual_nc_file = "epmsim/test_data/residual_output_test.nc"
    trend_nc_file = "epmsim/test_data/trend_output_test.nc"
    seasonal_nc_file = "epmsim/test_data/seasonal_output_test.nc"
    
    roi_lat_range = (28, 32)
    roi_lon_range = (172, 175)
    test_time =19  # Time index for trend data
    n_steps = 4
    period_length = 365
    save_dir = 'full_interpolation_results'
    true_sst_file ="epmsim/test_data/2006_days_roi.nc"  # True SST data file path

 
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load all data
    print("Loading data...")
    trend_data, lats, lons = load_roi_time_data(
        trend_nc_file, roi_lat_range, roi_lon_range,
        start_time=test_time-n_steps, end_time=test_time+n_steps,
        variable='trend'
    )
    
    seasonal_data, _, _ = load_roi_time_data(
        seasonal_nc_file, roi_lat_range, roi_lon_range,
        start_time=test_time-n_steps, end_time=test_time+n_steps,
        variable='seasonal'
    )


    residual_data, _, _ = load_roi_time_data(
        residual_nc_file, roi_lat_range, roi_lon_range,
        start_time=test_time-n_steps, end_time=test_time+n_steps,
        variable='residual'
    )


    # Initialize models
    print("Initializing models...")
    trend_interpolator = TrendInterpolator(trend_model_path)
    seasonal_interpolator = SeasonalInterpolator(seasonal_model_path)
    residual_interpolator =PSDTIM(spatial_window_size=3, max_displacement=3)
    # Perform interpolation
    print("Performing interpolation...")
    target_idx = n_steps
    
    # Trend interpolation
    predicted_trend = trend_interpolator.interpolate(trend_data)
    
    # Seasonal interpolation
    predicted_seasonal = seasonal_interpolator.interpolate(seasonal_data, test_time, period_length)

    start_time=int(test_time-4)
    end_time=int(test_time+3)
    print(type(start_time))
    print(type(end_time))


    # Residual interpolation
    predicted_residual,_,_,_ = residual_interpolator.interpolate(
        residual_data, 
        roi_lat_range, 
        roi_lon_range, 
        start_time, 
        end_time, 
        test_time,  
        window_size=17,     # Cross-correlation window size
        plot_results=False  # Whether to plot results
    )

    # Combine results
    predicted_combined = predicted_trend + predicted_seasonal + predicted_residual
    original_combined = (trend_data[target_idx] + 
                        seasonal_data[target_idx] + 
                        residual_data[target_idx])
    
    # Evaluate results
    metrics = {}
    components = {
        'trend': (predicted_trend, trend_data[target_idx]),
        'seasonal': (predicted_seasonal, seasonal_data[target_idx]),
        'residual': (predicted_residual, residual_data[target_idx]),
        'combined': (predicted_combined, original_combined)
    }
    
    for name, (pred, true) in components.items():
        component_metrics = evaluate_interpolation(pred, true)
        metrics[f"{name}_RMSE"] = component_metrics['RMSE']
        metrics[f"{name}_MAE"] = component_metrics['MAE']
    
    # Print evaluation results
    print("\nTrend model evaluation results:")
    print(f"RMSE: {metrics['trend_RMSE']:.6f}°C")
    print(f"MAE: {metrics['trend_MAE']:.6f}°C")
    
    print("\nSeasonal model evaluation results:")
    print(f"RMSE: {metrics['seasonal_RMSE']:.6f}°C")
    print(f"MAE: {metrics['seasonal_MAE']:.6f}°C")
    
    print("\nResidual model evaluation results:")
    print(f"RMSE: {metrics['residual_RMSE']:.6f}°C")
    print(f"MAE: {metrics['residual_MAE']:.6f}°C")
    
    print("\nCombined result evaluation:")
    print(f"RMSE: {metrics['combined_RMSE']:.6f}°C")
    print(f"MAE: {metrics['combined_MAE']:.6f}°C")
    true_sst=MY.get_sst_data_for_day_and_area(true_sst_file, test_time, roi_lat_range, roi_lon_range)
    # Call scatter plot function
    MY.plot_Scatter_plot_graph(true_sst, predicted_combined)

    # Visualize interpolated raster
    MY.plot_raster_grid(predicted_combined,roi_lat_range,roi_lon_range,15,22)



    MY.plot_raster_grid(true_sst,roi_lat_range,roi_lon_range,15,22)



if __name__ == '__main__':
    main()
