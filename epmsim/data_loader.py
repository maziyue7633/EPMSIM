import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import netCDF4 as nc

def load_roi_time_data(nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time):
    """
    Load SST data for a specified ROI and time range from a NetCDF file.

    Args:
        nc_file_path: Path to the NetCDF file
        roi_lat_range: Latitude range of the ROI (min_lat, max_lat)
        roi_lon_range: Longitude range of the ROI (min_lon, max_lon)
        start_time: Start time-step index
        end_time: End time-step index

    Returns:
        sst_roi_time: SST data within the ROI and time range (time, lat, lon)
        lats_roi: Latitude array for the ROI
        lons_roi: Longitude array for the ROI
    """
    ds = nc.Dataset(nc_file_path, 'r')
    
    # Extract latitude and longitude arrays
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    
    # Find indices for the ROI
    lat_min, lat_max = roi_lat_range
    lon_min, lon_max = roi_lon_range
    lat_min_idx = np.searchsorted(lats, lat_min)
    lat_max_idx = np.searchsorted(lats, lat_max)
    lon_min_idx = np.searchsorted(lons, lon_min)
    lon_max_idx = np.searchsorted(lons, lon_max)
    
    # Ensure time-step indices are valid
    num_times = len(ds.dimensions['time'])
    if start_time < 0 or end_time >= num_times:
        raise IndexError("start_time or end_time is out of range of time steps.")
    
    # Load SST data for the specified ROI and time range
    sst_var = ds.variables['analysed_sst']
    sst_roi_time = sst_var[
        start_time:end_time + 1,
        lat_min_idx:lat_max_idx,
        lon_min_idx:lon_max_idx
    ].astype(np.float32)
    
    # Convert from Kelvin to Celsius
    sst_roi_time = sst_roi_time - 273.15
    
    lats_roi = lats[lat_min_idx:lat_max_idx]
    lons_roi = lons[lon_min_idx:lon_max_idx]
    
    ds.close()
    return sst_roi_time, lats_roi, lons_roi

class SSTDataset(Dataset):
    def __init__(self, nc_file_path, roi_lat_range, roi_lon_range, 
                 start_time, end_time, n_steps):
        """
        Args:
            nc_file_path: Path to the NetCDF file
            roi_lat_range: Latitude range of the ROI (min_lat, max_lat)
            roi_lon_range: Longitude range of the ROI (min_lon, max_lon)
            start_time: Start time-step index
            end_time: End time-step index
            n_steps: Number of time steps before and after the target for interpolation
        """
        # Load data
        self.sst_data, self.lats, self.lons = load_roi_time_data(
            nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time
        )
        
        self.n_steps = n_steps
        
        # Normalize the data
        self.sst_normalized = self._normalize_data(self.sst_data)
        
        # Generate valid indices for interpolation (must have n_steps before and after)
        self.valid_indices = [
            i for i in range(n_steps, len(self.sst_normalized) - n_steps)
        ]
    
    def _normalize_data(self, data):
        """Normalize data to zero mean and unit variance."""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Input sequence containing n_steps before and after the target time step 
               (shape: 2*n_steps, height, width, 1)
            y: SST at the target time step (shape: height, width, 1)
        """
        center_idx = self.valid_indices[idx]
        
        # Get n_steps before
        x_before = self.sst_normalized[center_idx - self.n_steps:center_idx]
        
        # Get n_steps after
        x_after = self.sst_normalized[center_idx + 1:center_idx + self.n_steps + 1]
        
        # Target SST at the center time step
        y = self.sst_normalized[center_idx]
        
        # Concatenate before and after sequences
        x = np.concatenate([x_before, x_after], axis=0)
        
        # Convert to torch tensors and add channel dimension
        x = torch.FloatTensor(x)[..., np.newaxis]  # (2*n_steps, height, width, 1)
        y = torch.FloatTensor(y)[..., np.newaxis]  # (height, width, 1)
        
        return x, y

def get_dataloader(nc_file_path, roi_lat_range, roi_lon_range, 
                   start_time, end_time, n_steps, 
                   batch_size=32, val_split=0.2):
    """
    Create DataLoaders and split into training and validation sets.

    Args:
        nc_file_path: Path to the NetCDF file
        roi_lat_range: Latitude range of the ROI
        roi_lon_range: Longitude range of the ROI
        start_time: Start time-step index
        end_time: End time-step index
        n_steps: Number of time steps before and after for interpolation
        batch_size: Batch size (default: 32)
        val_split: Proportion of data to use for validation (default: 0.2)
    """
    # Instantiate the full dataset
    dataset = SSTDataset(
        nc_file_path=nc_file_path,
        roi_lat_range=roi_lat_range,
        roi_lon_range=roi_lon_range,
        start_time=start_time,
        end_time=end_time,
        n_steps=n_steps
    )
    
    # Determine sizes for train and validation sets
    total_size = len(dataset)
    train_size = int(total_size * (1 - val_split))
    
    # Split indices sequentially
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Parameter settings
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
    
    # Print number of batches
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Inspect one batch
    for x, y in train_loader:
        print("Input shape:", x.shape)
        print("Target shape:", y.shape)
        break
