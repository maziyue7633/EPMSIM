# test.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from conBiLSTM_model import create_model
from data_loader import get_dataloader
import os
from datetime import datetime
import netCDF4 as nc




def load_roi_time_data(nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time):
    """
    Load SST data for a specified ROI and time range from a NetCDF file.

    Args:
        nc_file_path (str): Path to the NetCDF file.
        roi_lat_range (tuple): Latitude range of the ROI (min_lat, max_lat).
        roi_lon_range (tuple): Longitude range of the ROI (min_lon, max_lon).
        start_time (int): Start time-step index.
        end_time (int): End time-step index.

    Returns:
        sst_roi_time (ndarray): SST data within the ROI and time range (time, lat, lon) in °C.
        lats_roi (ndarray): Latitude array for the ROI.
        lons_roi (ndarray): Longitude array for the ROI.
    """
    ds = nc.Dataset(nc_file_path, 'r')
    
    # Read latitude and longitude arrays
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    
    # Find the index bounds for the ROI
    lat_min, lat_max = roi_lat_range
    lon_min, lon_max = roi_lon_range
    lat_min_idx = np.searchsorted(lats, lat_min)
    lat_max_idx = np.searchsorted(lats, lat_max)
    lon_min_idx = np.searchsorted(lons, lon_min)
    lon_max_idx = np.searchsorted(lons, lon_max)
    
    # Validate time-step indices
    num_times = len(ds.dimensions['time'])
    if start_time < 0 or end_time >= num_times:
        raise IndexError("start_time or end_time is out of range of time steps.")
    
    # Load SST variable and slice to ROI and time range
    sst_var = ds.variables['analysed_sst']
    sst_roi_time = sst_var[
        start_time:end_time + 1,
        lat_min_idx:lat_max_idx,
        lon_min_idx:lon_max_idx
    ].astype(np.float32)
    
    # Convert Kelvin to Celsius
    sst_roi_time = sst_roi_time - 273.15
    
    lats_roi = lats[lat_min_idx:lat_max_idx]
    lons_roi = lons[lon_min_idx:lon_max_idx]
    
    ds.close()
    return sst_roi_time, lats_roi, lons_roi

class SSTInterpolator:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the SSTInterpolator with a trained model.

        Args:
            model_path (str): Path to the saved model checkpoint.
            device (str): Preferred device ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model and load weights
        self.model = create_model(input_channels=1, hidden_channels=12, kernel_size=3)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.best_val_loss = checkpoint.get('val_loss', None)
        if self.best_val_loss is not None:
            print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def prepare_interpolation_data(self, sst_data):
        """
        Prepare data for interpolation by normalizing and formatting.

        Args:
            sst_data (ndarray): Raw SST data array.

        Returns:
            x (Tensor): Input tensor for the model.
            mean (float): Mean of the original data.
            std (float): Standard deviation of the original data.
        """
        # Normalize data
        mean = np.mean(sst_data)
        std = np.std(sst_data)
        sst_normalized = (sst_data - mean) / std
        
        # Format for model input
        x = torch.FloatTensor(sst_normalized)[None, :, :, :, None]  # [1, time, H, W, 1]
        return x, mean, std
    
    def interpolate(self, sst_data):
        """
        Perform interpolation using the loaded model.

        Args:
            sst_data (ndarray): SST data to interpolate.

        Returns:
            predicted (ndarray): Interpolated SST in °C.
        """
        x, mean, std = self.prepare_interpolation_data(sst_data)
        x = x.to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
        
        # Denormalize and squeeze extra dimensions
        predicted = output.cpu().numpy().squeeze() * std + mean
        return predicted

    def evaluate_interpolation(self, predicted, true_value):
        """
        Evaluate interpolation results with RMSE, MAE, and MAPE.

        Args:
            predicted (ndarray): Interpolated SST array.
            true_value (ndarray): True SST array.

        Returns:
            dict: {'RMSE', 'MAE', 'MAPE'}.
        """
        # Ensure matching dimensions
        pred = predicted.squeeze()
        true = true_value.squeeze()
        
        # Compute metrics
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred - true))
        
        # Compute MAPE (avoid division by zero)
        epsilon = 1e-10
        mape = np.mean(np.abs((true - pred) / (true + epsilon))) * 100
        
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def plot_results(original, predicted, lats, lons, metrics, save_dir):
    """
    Plot and save original vs. interpolated SST.

    Args:
        original (ndarray): Original SST array.
        predicted (ndarray): Interpolated SST array.
        lats (ndarray): Latitude array.
        lons (ndarray): Longitude array.
        metrics (dict): Evaluation metrics.
        save_dir (str): Directory to save the figure.

    Returns:
        str: Path to the saved figure.
    """
    orig = original.squeeze()
    pred = predicted.squeeze()
    
    # Determine color scale limits
    vmin = min(orig.min(), pred.min())
    vmax = max(orig.max(), pred.max())
    
    plt.figure(figsize=(15, 5))
    
    # Plot original SST
    plt.subplot(1, 2, 1)
    plt.pcolormesh(lons, lats, orig, shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Temperature (°C)')
    plt.title('Original SST')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Plot interpolated SST
    plt.subplot(1, 2, 2)
    plt.pcolormesh(lons, lats, pred, shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Temperature (°C)')
    title = (f'Interpolated SST\nRMSE: {metrics["RMSE"]:.4f}°C\n'
             f'MAE: {metrics["MAE"]:.4f}°C\n'
             f'MAPE: {metrics["MAPE"]:.4f}%')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'interpolation_results_{timestamp}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return save_path

def calculate_linear_interpolation(sst_data, target_idx):
    """
    Compute linear interpolation between two surrounding time steps.

    Args:
        sst_data (ndarray): SST data array.
        target_idx (int): Index of the target time step.

    Returns:
        ndarray: Linearly interpolated SST.
    """
    before = sst_data[target_idx - 1]
    after  = sst_data[target_idx + 1]
    return (before + after) / 2

def main():
    # Parameter settings
    model_path         = 'checkpoints_test/best_model_convBiLSTM_day.pth'
    nc_file            = "2006-2008-merged_data.nc"       # Data for prediction
    true_nc_file       = "epmsim/test_data/merged_data.nc" # True SST dataset
    roi_lat_range      = (28, 32)
    roi_lon_range      = (172, 175)
    test_time          = 29    # Index in the prediction dataset
    true_test_time     = 29    # Corresponding index in the true dataset
    n_steps            = 4     # Must match training configuration
    save_dir           = 'results_test'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data for prediction
    print("Loading data for prediction...")
    sst_data, lats, lons = load_roi_time_data(
        nc_file, roi_lat_range, roi_lon_range,
        start_time=test_time - n_steps,
        end_time=test_time + n_steps
    )
    
    # Load true SST data
    print("Loading true data...")
    true_sst_data, _, _ = load_roi_time_data(
        true_nc_file, roi_lat_range, roi_lon_range,
        start_time=true_test_time,
        end_time=true_test_time
    )
   
    # Initialize interpolator
    print("Initializing model...")
    interpolator = SSTInterpolator(model_path)
    
    # Perform interpolation
    print("Performing interpolation...")
    predicted_sst = interpolator.interpolate(sst_data)
    
    # Compute linear interpolation baseline
    linear_sst = calculate_linear_interpolation(sst_data, n_steps)
    
    # Evaluate against true values
    model_metrics  = interpolator.evaluate_interpolation(predicted_sst, true_sst_data[0])
    linear_metrics = interpolator.evaluate_interpolation(linear_sst,      true_sst_data[0])
    
    print("\nModel evaluation results (compared to true values):")
    print(f"RMSE: {model_metrics['RMSE']:.6f}°C, MAE: {model_metrics['MAE']:.6f}°C, MAPE: {model_metrics['MAPE']:.4f}%")
    
    print("\nLinear interpolation evaluation results (compared to true values):")
    print(f"RMSE: {linear_metrics['RMSE']:.6f}°C, MAE: {linear_metrics['MAE']:.6f}°C, MAPE: {linear_metrics['MAPE']:.4f}%")
    
    # Plot and save results
    print("Saving results...")
    plot_path = plot_results(true_sst_data[0], predicted_sst, lats, lons, model_metrics, save_dir)
    print(f"Results saved to: {plot_path}")

if __name__ == '__main__':
    main()



 