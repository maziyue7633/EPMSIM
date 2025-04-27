import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import netCDF4 as nc
import pickle
import os

def load_roi_time_data(nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time):
    """
    Load SST data from a NetCDF file for a specified ROI and time range.

    Args:
        nc_file_path (str): Path to the NetCDF file.
        roi_lat_range (tuple): Latitude range of the ROI (min_lat, max_lat).
        roi_lon_range (tuple): Longitude range of the ROI (min_lon, max_lon).
        start_time (int): Start time-step index.
        end_time (int): End time-step index.

    Returns:
        sst_roi_time (ndarray): SST data in the ROI and time range (time, lat, lon).
        lats_roi (ndarray): Latitude array for the ROI.
        lons_roi (ndarray): Longitude array for the ROI.
    """
    ds = nc.Dataset(nc_file_path, 'r')
    
    # Extract latitude and longitude arrays
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    
    # Determine index bounds for the ROI
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
    
    # Load the 'trend' variable for the ROI and time range
    sst_var = ds.variables['trend']
    sst_roi_time = sst_var[
        start_time:end_time + 1,
        lat_min_idx:lat_max_idx,
        lon_min_idx:lon_max_idx
    ].astype(np.float32)
    
    # No conversion from Kelvin needed
    # sst_roi_time = sst_roi_time
    
    lats_roi = lats[lat_min_idx:lat_max_idx]
    lons_roi = lons[lon_min_idx:lon_max_idx]
    
    ds.close()
    return sst_roi_time, lats_roi, lons_roi

def save_train_stats(sst_mean, sst_std, save_dir):
    """
    Save training data mean and standard deviation to a pickle file.

    Args:
        sst_mean (float): Mean of the training SST data.
        sst_std (float): Standard deviation of the training SST data.
        save_dir (str): Directory where the stats file will be saved.
    """
    file_path = os.path.join(save_dir, 'train_stats.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump((sst_mean, sst_std), f)

class SSTDataset(Dataset):
    def __init__(self, nc_file_path, roi_lat_range, roi_lon_range, 
                 start_time, end_time, n_steps):
        """
        Dataset for SST interpolation.

        Args:
            nc_file_path (str): Path to the NetCDF file.
            roi_lat_range (tuple): Latitude range of the ROI.
            roi_lon_range (tuple): Longitude range of the ROI.
            start_time (int): Start time-step index.
            end_time (int): End time-step index.
            n_steps (int): Number of time steps before and after the target.
        """
        # Load data
        self.sst_data, self.lats, self.lons = load_roi_time_data(
            nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time
        )
        
        self.n_steps = n_steps
        
        # Normalize data
        self.sst_normalized = self._normalize_data(self.sst_data)
        
        # Build valid indices ensuring n_steps before and after
        self.valid_indices = [
            i for i in range(n_steps, len(self.sst_normalized) - n_steps)
        ]
    
    def _normalize_data(self, data):
        """Normalize data to zero mean and unit variance, and save stats."""
        mean = np.mean(data)
        std = np.std(data)
        save_train_stats(mean, std, 'trend_checkpoints_test')
        return (data - mean) / std
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a sample for interpolation.

        Returns:
            x (Tensor): Input sequence of shape (2*n_steps, height, width, 1).
            y (Tensor): SST at the target time step of shape (height, width, 1).
        """
        center_idx = self.valid_indices[idx]
        
        # Get previous n_steps
        x_before = self.sst_normalized[center_idx - self.n_steps:center_idx]
        # Get next n_steps
        x_after = self.sst_normalized[center_idx + 1:center_idx + self.n_steps + 1]
        # Target SST
        y = self.sst_normalized[center_idx]
        
        # Concatenate before and after sequences
        x = np.concatenate([x_before, x_after], axis=0)
        
        # Convert to torch tensors with channel dimension
        x = torch.FloatTensor(x)[..., np.newaxis]  # (2*n_steps, height, width, 1)
        y = torch.FloatTensor(y)[..., np.newaxis]  # (height, width, 1)
        
        return x, y

def get_dataloader(nc_file_path, roi_lat_range, roi_lon_range, 
                   start_time, end_time, n_steps, 
                   batch_size=32, val_split=0.2):
    """
    Create DataLoaders for training and validation.

    Args:
        nc_file_path (str): Path to the NetCDF file.
        roi_lat_range (tuple): Latitude range of the ROI.
        roi_lon_range (tuple): Longitude range of the ROI.
        start_time (int): Start time-step index.
        end_time (int): End time-step index.
        n_steps (int): Number of time steps before and after the target.
        batch_size (int): Batch size (default: 32).
        val_split (float): Fraction of data for validation (default: 0.2).

    Returns:
        train_loader (DataLoader), val_loader (DataLoader)
    """
    dataset = SSTDataset(
        nc_file_path=nc_file_path,
        roi_lat_range=roi_lat_range,
        roi_lon_range=roi_lon_range,
        start_time=start_time,
        end_time=end_time,
        n_steps=n_steps
    )
    
    total_size = len(dataset)
    train_size = int(total_size * (1 - val_split))
    
    # Sequential split
    train_indices = list(range(train_size))
    val_indices   = list(range(train_size, total_size))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    nc_file = "mync_interval_2.nc"
    roi_lat_range = (28, 32)
    roi_lon_range = (172, 175)
    start_time = 0
    end_time = 100
    n_steps = 2
    batch_size = 32
    
    train_loader, val_loader = get_dataloader(
        nc_file_path=nc_file,
        roi_lat_range=roi_lat_range,
        roi_lon_range=roi_lon_range,
        start_time=start_time,
        end_time=end_time,
        n_steps=n_steps,
        batch_size=batch_size
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    for x, y in train_loader:
        print("Input shape:", x.shape)
        print("Target shape:", y.shape)
        break
