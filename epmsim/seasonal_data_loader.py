import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import netCDF4 as nc

def load_roi_time_data(nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time):
    """
    Load SST data for a specified ROI and time range from a NetCDF file.

    Args:
        nc_file_path (str): Path to the NetCDF file
        roi_lat_range (tuple): Latitude range of the ROI (min_lat, max_lat)
        roi_lon_range (tuple): Longitude range of the ROI (min_lon, max_lon)
        start_time (int): Start time-step index
        end_time (int): End time-step index

    Returns:
        sst_roi_time (ndarray): SST data within the ROI and time range (time, lat, lon)
        lats_roi (ndarray): Latitude array for the ROI
        lons_roi (ndarray): Longitude array for the ROI
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
    
    # Validate time-step indices
    num_times = len(ds.dimensions['time'])
    if start_time < 0 or end_time >= num_times:
        raise IndexError("start_time or end_time is out of range of time steps.")
    
    # Load SST data for the specified ROI and time range
    sst_var = ds.variables['seasonal']
    sst_roi_time = sst_var[
        start_time:end_time + 1,
        lat_min_idx:lat_max_idx,
        lon_min_idx:lon_max_idx
    ].astype(np.float32)
    
    # Convert to Celsius (do not subtract 273.15)
    # sst_roi_time = sst_roi_time
    
    lats_roi = lats[lat_min_idx:lat_max_idx]
    lons_roi = lons[lon_min_idx:lon_max_idx]
    
    ds.close()
    return sst_roi_time, lats_roi, lons_roi

class SSTDataset(Dataset):
    def __init__(self, nc_file_path, roi_lat_range, roi_lon_range, 
                 start_time, end_time, n_steps, period_length):
        """
        Args:
            nc_file_path (str): Path to the NetCDF file
            roi_lat_range (tuple): Latitude range of the ROI (min_lat, max_lat)
            roi_lon_range (tuple): Longitude range of the ROI (min_lon, max_lon)
            start_time (int): Start time-step index
            end_time (int): End time-step index
            n_steps (int): Number of time steps before and after the target for interpolation
            period_length (int): Length of the period used to compute periodic positional features
        """
        # Load data
        self.sst_data, self.lats, self.lons = load_roi_time_data(
            nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time
        )
        
        self.n_steps = n_steps
        self.period_length = period_length
        
        # Normalize data
        self.sst_normalized = self._normalize_data(self.sst_data)
        
        # Generate valid target indices (ensure n_steps before and after exist)
        self.valid_indices = [
            i for i in range(n_steps, len(self.sst_normalized) - n_steps)
        ]
    
    def _normalize_data(self, data):
        """Normalize data to zero mean and unit variance."""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    
    def _compute_periodic_features(self, idx):
        """
        Compute periodic positional features (sine and cosine).

        Args:
            idx (int): Current time-step index

        Returns:
            sin_feat (float): Sine feature
            cos_feat (float): Cosine feature
        """
        phase = 2 * np.pi * idx / self.period_length
        sin_feat = np.sin(phase)
        cos_feat = np.cos(phase)
        return sin_feat, cos_feat
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            x (Tensor): Input sequence with shape [time_steps, height, width, channels=3]
            y (Tensor): SST at the target time step with shape [height, width, 1]
        """
        center_idx = self.valid_indices[idx]
        
        # Extract SST for time steps before and after the target
        x_before = self.sst_normalized[center_idx - self.n_steps:center_idx]  # [n_steps, H, W]
        x_after  = self.sst_normalized[center_idx + 1:center_idx + self.n_steps + 1]  # [n_steps, H, W]
        
        # Target SST
        y = self.sst_normalized[center_idx]  # [H, W]
        
        # Combine before and after sequences
        x_combined = np.concatenate([x_before, x_after], axis=0)  # [2*n_steps, H, W]
        
        # Compute periodic features
        sin_feat, cos_feat = self._compute_periodic_features(center_idx)
        
        # Prepare input tensor [time_steps, H, W, 3]
        time_steps, height, width = x_combined.shape
        x = np.zeros((time_steps, height, width, 3), dtype=np.float32)
        x[..., 0] = x_combined    # SST data channel
        x[..., 1] = sin_feat      # sine feature channel
        x[..., 2] = cos_feat      # cosine feature channel
        
        # Convert to torch tensors
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)[..., np.newaxis]  # [H, W, 1]
        
        return x, y

def get_dataloader(nc_file_path, roi_lat_range, roi_lon_range, 
                   start_time, end_time, n_steps, period_length,
                   batch_size=32, val_split=0.2):
    """
    Create DataLoaders and split into training and validation sets.

    Args:
        nc_file_path (str): Path to the NetCDF file
        roi_lat_range (tuple): Latitude range of the ROI
        roi_lon_range (tuple): Longitude range of the ROI
        start_time (int): Start time-step index
        end_time (int): End time-step index
        n_steps (int): Number of time steps before and after for interpolation
        period_length (int): Length of the period for periodic features
        batch_size (int): Batch size (default: 32)
        val_split (float): Fraction of data for validation (default: 0.2)

    Returns:
        train_loader (DataLoader), val_loader (DataLoader)
    """
    # Instantiate dataset
    dataset = SSTDataset(
        nc_file_path=nc_file_path,
        roi_lat_range=roi_lat_range,
        roi_lon_range=roi_lon_range,
        start_time=start_time,
        end_time=end_time,
        n_steps=n_steps,
        period_length=period_length
    )
    
    total_size = len(dataset)
    train_size = int(total_size * (1 - val_split))
    
    # Split indices sequentially
    train_indices = list(range(train_size))
    val_indices   = list(range(train_size, total_size))
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
