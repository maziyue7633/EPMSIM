import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, map_coordinates
import warnings
from tqdm import tqdm
import numpy.fft as fft
import gc
import os

import sys
# Add parent directory to system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import myfunction as MY

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity as ssim

class PSDTIM:
    """
    Path-based spatiotemporal IDW method for SST interpolation.
    This class implements particle tracking and spatiotemporal IDW interpolation.
    """
    
    def __init__(self, spatial_window_size=3, max_displacement=5, idw_power=2, min_distance=1e-10):
        """
        Initialize the PSDTIM class.

        Parameters:
        - spatial_window_size: window size for spatial IDW interpolation (default=3)
        - max_displacement: maximum allowed displacement of grid points (default=5)
        - idw_power: power parameter for IDW calculation (default=2)
        - min_distance: minimum distance threshold to prevent division by zero (default=1e-10)
        """
        self.spatial_window_size = spatial_window_size
        self.max_displacement = max_displacement
        self.idw_power = idw_power
        self.min_distance = min_distance
    
    def load_roi_time_data(self, nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time):
        """
        Load SST data for a specified ROI and time range from a NetCDF file.

        Parameters:
        - nc_file_path: Path to the NetCDF file
        - roi_lat_range: Latitude range of the ROI (min_lat, max_lat)
        - roi_lon_range: Longitude range of the ROI (min_lon, max_lon)
        - start_time: Start time-step index (0-based)
        - end_time: End time-step index (0-based, inclusive)

        Returns:
        - sst_roi_time: SST data within the ROI and time range (time, lat, lon)
        - lats_roi: Latitude array for the ROI
        - lons_roi: Longitude array for the ROI
        """
        ds = nc.Dataset(nc_file_path, 'r')
        
        # Extract latitude and longitude arrays
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        
        # Determine ROI indices
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
        sst_var = ds.variables['analysed_sst']
        sst_roi_time = sst_var[
            start_time:end_time + 1,
            lat_min_idx:lat_max_idx,
            lon_min_idx:lon_max_idx
        ].astype(np.float32)
        
        # Handle missing values
        fill_value = getattr(sst_var, '_FillValue', None)
        if fill_value is not None:
            sst_roi_time = np.where(sst_roi_time == fill_value, np.nan, sst_roi_time)
        else:
            sst_roi_time = np.where(np.isnan(sst_roi_time), np.nan, sst_roi_time)
        
        # Convert from Kelvin to Celsius
        sst_roi_time -= 273.15
        
        lats_roi = lats[lat_min_idx:lat_max_idx]
        lons_roi = lons[lon_min_idx:lon_max_idx]
        
        ds.close()
        return sst_roi_time, lats_roi, lons_roi

    def fft_cross_correlation(self, arr1, arr2):
        """
        Perform 2D cross-correlation using FFT.

        Parameters:
        - arr1: First 2D array
        - arr2: Second 2D array

        Returns:
        - cross_corr: 2D array of the cross-correlation result
        """
        f1 = fft.fft2(arr1)
        f2 = fft.fft2(np.flipud(np.fliplr(arr2)))
        cross_corr = fft.ifft2(f1 * f2)
        cross_corr = np.real(cross_corr)
        cross_corr = fft.fftshift(cross_corr)  # Move zero-frequency component to center
        return cross_corr
    
    def compute_displacement_fft(self, arr1, arr2, window_size=None, max_displacement=None):
        """
        Compute displacement vectors between two time steps using FFT-based cross-correlation.

        Parameters:
        - arr1: SST data at the first time step (2D array)
        - arr2: SST data at the second time step (2D array)
        - window_size: Size of the correlation window (must be odd)
        - max_displacement: Maximum allowed displacement (in grid points)

        Returns:
        - displacement_y: Vertical displacement vector (2D array)
        - displacement_x: Horizontal displacement vector (2D array)
        """
        if window_size is None:
            window_size = 7
        if max_displacement is None:
            max_displacement = self.max_displacement
            
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        
        half_win = window_size // 2
        shape = arr1.shape
        displacement_y = np.zeros(shape, dtype=np.float32)
        displacement_x = np.zeros(shape, dtype=np.float32)
        
        # Function to fill NaNs via local averaging
        def fill_nan(arr, size=3):
            nan_mask = np.isnan(arr)
            arr_filled = uniform_filter(np.nan_to_num(arr, nan=0.0), size=size, mode='constant')
            if np.sum(~nan_mask) == 0:
                return np.zeros_like(arr)
            arr_filled[nan_mask] = np.nanmean(arr)
            return arr_filled
        
        sst1_filled = fill_nan(arr1, size=window_size//3)
        sst2_filled = fill_nan(arr2, size=window_size//3)
        
        # Slide window over each grid point
        for i in tqdm(range(half_win, shape[0] - half_win), desc="Computing displacement vectors (rows)"):
            for j in range(half_win, shape[1] - half_win):
                window1 = sst1_filled[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1]
                window2 = sst2_filled[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1]
                
                # Normalize windows
                w1_mean, w1_std = np.nanmean(window1), np.nanstd(window1)
                w2_mean, w2_std = np.nanmean(window2), np.nanstd(window2)
                if w1_std == 0 or w2_std == 0:
                    continue
                window1_norm = (window1 - w1_mean) / w1_std
                window2_norm = (window2 - w2_mean) / w2_std
                
                # Cross-correlation via FFT
                correlation = self.fft_cross_correlation(window1_norm, window2_norm)
                
                # Find peak location
                y_center, x_center = np.array(correlation.shape) // 2
                y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)
                
                dy = y_peak - y_center
                dx = x_peak - x_center
                
                # Discard outliers
                if abs(dy) > max_displacement or abs(dx) > max_displacement:
                    continue
                
                displacement_y[i, j] = dy
                displacement_x[i, j] = dx
        
        return displacement_y, displacement_x
    
    def idw(self, center_value, neighbors, power=None, min_distance=None):
        """
        Inverse Distance Weighting (IDW) interpolation with overflow protection.

        Parameters:
        - center_value: Value at the center point
        - neighbors: List of (value, distance) tuples for neighboring points
        - power: Power parameter for IDW (default uses instance value)
        - min_distance: Minimum distance threshold (default uses instance value)

        Returns:
        - Interpolated value
        """
        if power is None:
            power = self.idw_power
        if min_distance is None:
            min_distance = self.min_distance
            
        weights_sum = 0.0
        weighted_value_sum = 0.0
        for value, distance in neighbors:
            if distance < min_distance:
                continue
            weight = 1.0 / (distance ** power)
            weights_sum += weight
            weighted_value_sum += value * weight
        
        if weights_sum == 0.0:
            return center_value
        result = 0.5 * center_value + 0.5 * (weighted_value_sum / weights_sum)
        return round(result, 2)
    
    def window_operation(self, grid, i, j, spatial_window_size=None):
        """
        Apply IDW interpolation over a local spatial window.

        Parameters:
        - grid: Input grid
        - i: Row index
        - j: Column index
        - spatial_window_size: Spatial window size (default uses instance value)

        Returns:
        - Interpolated value at (i, j)
        """
        if spatial_window_size is None:
            spatial_window_size = self.spatial_window_size
            
        radius = spatial_window_size // 2
        center_value = grid[i, j]
        neighbors = []
        
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    distance = np.hypot(di, dj)
                    neighbors.append((grid[ni, nj], distance))
                    
        return self.idw(center_value, neighbors)
    
    def apply_window_idw(self, grid, spatial_window_size=None):
        """
        Smooth an entire grid by applying local IDW interpolation.

        Parameters:
        - grid: Input grid
        - spatial_window_size: Spatial window size (default uses instance value)

        Returns:
        - Smoothed grid
        """
        if spatial_window_size is None:
            spatial_window_size = self.spatial_window_size
            
        smoothed_grid = grid.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                smoothed_grid[i, j] = self.window_operation(grid, i, j, spatial_window_size)
        return smoothed_grid
    
    def time_idw(self, sst_values, time_points, t_float, power=2, min_time_distance=0.001):
        """
        Perform time-based IDW interpolation.

        Parameters:
        - sst_values: List of SST values at different time points
        - time_points: Corresponding list of time points
        - t_float: Target interpolation time
        - power: Power parameter for IDW
        - min_time_distance: Minimum time distance threshold

        Returns:
        - Interpolated SST value at t_float
        """
        if not sst_values or not time_points:
            return np.nan
        if len(sst_values) != len(time_points):
            raise ValueError("sst_values and time_points must have the same length.")
        
        # Filter out NaNs
        valid = [(s, t) for s, t in zip(sst_values, time_points) if not np.isnan(s)]
        if not valid:
            return np.nan
        
        weights_sum = 0.0
        weighted_sum = 0.0
        
        for sst, t in valid:
            dt = abs(t - t_float)
            if dt < min_time_distance:
                return sst
            w = 1.0 / (dt ** power)
            weights_sum += w
            weighted_sum += sst * w
        
        return weighted_sum / weights_sum if weights_sum else np.nan
    
    def track_forward(self, sst_roi_time, displacement_list_y, displacement_list_x,
                      t_start_idx, t_end_idx, grid_i, grid_j, start_time):
        """
        Track a grid point forward in time and collect SST values.

        Parameters:
        - sst_roi_time: SST data (time, lat, lon)
        - displacement_list_y: Vertical displacement vectors
        - displacement_list_x: Horizontal displacement vectors
        - t_start_idx: Relative start time index
        - t_end_idx: Relative end time index
        - grid_i: Starting row index
        - grid_j: Starting column index
        - start_time: Absolute start time index

        Returns:
        - sst_values_forward: List of tracked SST values going forward
        - time_points_forward: Corresponding list of absolute time points
        - count_forward: Number of points tracked forward
        """
        sst_values_forward = []
        time_points_forward = []
        curr_i, curr_j = float(grid_i), float(grid_j)
        
        # Record initial SST
        sst_initial = sst_roi_time[t_start_idx, grid_i, grid_j]
        if not np.isnan(sst_initial):
            sst_values_forward.append(sst_initial)
            time_points_forward.append(start_time + t_start_idx)
        
        for t in range(t_start_idx, t_end_idx):
            dy = displacement_list_y[t, int(round(curr_i)), int(round(curr_j))]
            dx = displacement_list_x[t, int(round(curr_i)), int(round(curr_j))]
            
            if dy != 0 or dx != 0:
                curr_i += dy
                curr_j += dx
                curr_i = np.clip(curr_i, 0, sst_roi_time.shape[1] - 1)
                curr_j = np.clip(curr_j, 0, sst_roi_time.shape[2] - 1)
            
            sst_tracked = map_coordinates(
                sst_roi_time[t + 1],
                [[curr_i], [curr_j]],
                order=1,
                mode='nearest'
            )[0]
            
            if not np.isnan(sst_tracked):
                sst_values_forward.append(sst_tracked)
                time_points_forward.append(start_time + t + 1)
        
        return sst_values_forward, time_points_forward, len(sst_values_forward)
    
    def track_backward(self, sst_roi_time, displacement_list_y, displacement_list_x,
                       t_start_idx, t_end_idx, grid_i, grid_j, start_time):
        """
        Track a grid point backward in time and collect SST values.

        Parameters:
        - sst_roi_time: SST data (time, lat, lon)
        - displacement_list_y: Vertical displacement vectors
        - displacement_list_x: Horizontal displacement vectors
        - t_start_idx: Relative start time index
        - t_end_idx: Relative end time index (usually 0)
        - grid_i: Starting row index
        - grid_j: Starting column index
        - start_time: Absolute start time index

        Returns:
        - sst_values_backward: List of tracked SST values going backward
        - time_points_backward: Corresponding list of absolute time points
        - count_backward: Number of points tracked backward
        """
        sst_values_backward = []
        time_points_backward = []
        curr_i, curr_j = float(grid_i), float(grid_j)
        
        sst_initial = sst_roi_time[t_start_idx, grid_i, grid_j]
        if not np.isnan(sst_initial):
            sst_values_backward.append(sst_initial)
            time_points_backward.append(start_time + t_start_idx)
        
        for t in range(t_start_idx - 1, t_end_idx - 1, -1):
            dy = displacement_list_y[t, int(round(curr_i)), int(round(curr_j))]
            dx = displacement_list_x[t, int(round(curr_i)), int(round(curr_j))]
            
            if dy != 0 or dx != 0:
                curr_i -= dy
                curr_j -= dx
                curr_i = np.clip(curr_i, 0, sst_roi_time.shape[1] - 1)
                curr_j = np.clip(curr_j, 0, sst_roi_time.shape[2] - 1)
            
            sst_tracked = map_coordinates(
                sst_roi_time[t],
                [[curr_i], [curr_j]],
                order=1,
                mode='nearest'
            )[0]
            
            if not np.isnan(sst_tracked):
                sst_values_backward.append(sst_tracked)
                time_points_backward.append(start_time + t)
        
        return sst_values_backward, time_points_backward, len(sst_values_backward)
    
    def track_and_interpolate_sst(self, sst_roi_time, displacement_list_y, displacement_list_x,
                                  roi_lat_range, roi_lon_range, t_float, start_time, end_time):
        """
        Track each grid point and interpolate SST using spatiotemporal IDW.

        Parameters:
        - sst_roi_time: SST data (time, lat, lon)
        - displacement_list_y: Vertical displacement vectors (num_steps, lat, lon)
        - displacement_list_x: Horizontal displacement vectors
        - roi_lat_range: Latitude range of the ROI
        - roi_lon_range: Longitude range of the ROI
        - t_float: Floating-point time for interpolation
        - start_time: Absolute start time index
        - end_time: Absolute end time index

        Returns:
        - interp_sst: 2D array of interpolated SST values
        - moved_points_count: Total number of displaced grid points
        - count_grid_forward: 2D array of forward-tracked point counts
        - count_grid_backward: 2D array of backward-tracked point counts
        """
        num_steps = displacement_list_y.shape[0]
        nlat, nlon = sst_roi_time.shape[1:]
        interp_sst = np.full((nlat, nlon), np.nan, dtype=np.float32)
        moved_points_count = 0
        count_grid_forward = np.zeros((nlat, nlon), dtype=int)
        count_grid_backward = np.zeros((nlat, nlon), dtype=int)
        
        t_prev = max(start_time, int(np.floor(t_float)))
        t_next = min(end_time, int(np.ceil(t_float)))
        t_prev_idx = t_prev - start_time
        t_next_idx = t_next - start_time
        
        for i in tqdm(range(nlat), desc="Tracking and interpolating grid points (rows)"):
            for j in range(nlon):
                # Forward tracking
                if t_next_idx < num_steps:
                    sf, tf, cf = self.track_forward(
                        sst_roi_time, displacement_list_y, displacement_list_x,
                        t_start_idx=t_next_idx, t_end_idx=num_steps,
                        grid_i=i, grid_j=j, start_time=start_time
                    )
                else:
                    sf, tf, cf = [sst_roi_time[t_next_idx-1, i, j]], [start_time+t_next_idx-1], 1
                
                # Backward tracking
                if t_prev_idx > 0:
                    sb, tb, cb = self.track_backward(
                        sst_roi_time, displacement_list_y, displacement_list_x,
                        t_start_idx=t_prev_idx, t_end_idx=0,
                        grid_i=i, grid_j=j, start_time=start_time
                    )
                else:
                    sb, tb, cb = [sst_roi_time[t_prev_idx, i, j]], [start_time+t_prev_idx], 1
                
                if cf > 1 or cb > 1:
                    moved_points_count += 1
                    
                    # Spatial smoothing of tracked values
                    smoothed_f = []
                    for idx_v, val in enumerate(sf):
                        temp = np.full((3,3), val)
                        smoothed_f.append(self.window_operation(temp, 1, 1))
                    smoothed_b = []
                    for val in sb:
                        temp = np.full((3,3), val)
                        smoothed_b.append(self.window_operation(temp, 1, 1))
                    
                    all_vals = smoothed_b + smoothed_f
                    all_times = tb + tf
                    interp_sst[i, j] = self.time_idw(all_vals, all_times, t_float)
                    count_grid_forward[i, j] = cf
                    count_grid_backward[i, j] = cb
        
        return interp_sst, moved_points_count, count_grid_forward, count_grid_backward
    
    def plot_interpolated_sst(self, interp_sst, Lon, Lat, roi_lon_range, roi_lat_range, vmin=15, vmax=22):
        """Plot the interpolated SST image."""
        plt.figure(figsize=(12, 8))
        im = plt.imshow(
            interp_sst,
            extent=[roi_lon_range[0], roi_lon_range[1], roi_lat_range[0], roi_lat_range[1]],
            origin='lower', cmap='jet', vmin=vmin, vmax=vmax
        )
        plt.title('Interpolated SST at Floating-Point Time')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(im, label='Interpolated SST (°C)')
        plt.tight_layout()
        plt.show()
    
    def interpolate(self, sst_roi_time, roi_lat_range, roi_lon_range,
                    start_time, end_time, t_float, window_size=7,
                    max_displacement=None, plot_results=True):
        """
        Run the full PSDTIM workflow.

        Parameters:
        - sst_roi_time: SST data (time, lat, lon)
        - roi_lat_range: Latitude range of the ROI (min_lat, max_lat)
        - roi_lon_range: Longitude range of the ROI (min_lon, max_lon)
        - start_time: Start time-step index (0-based)
        - end_time: End time-step index (0-based, inclusive)
        - t_float: Floating-point time for interpolation
        - window_size: Correlation window size
        - max_displacement: Maximum allowed displacement
        - plot_results: Whether to plot the results

        Returns:
        - interp_sst: 2D array of interpolated SST values
        - moved_points_count: Total number of displaced grid points
        - count_grid_forward: 2D array of forward-tracked point counts
        - count_grid_backward: 2D array of backward-tracked point counts
        """
        gc.collect()
        warnings.filterwarnings("ignore")
        
        if isinstance(t_float, int):
            t_float = t_float - 0.5
        
        if max_displacement is None:
            max_displacement = self.max_displacement
        
        print("Loading SST data for specified ROI and time range...")
        # (sst_roi_time, _, _) = self.load_roi_time_data(...)
        
        # Compute displacement vectors between consecutive time steps
        disp_y_list, disp_x_list = [], []
        for t in range(end_time - start_time):
            print(f"Computing displacement from step {start_time + t} to {start_time + t + 1}...")
            dy, dx = self.compute_displacement_fft(
                sst_roi_time[t], sst_roi_time[t + 1],
                window_size=window_size, max_displacement=max_displacement
            )
            disp_y_list.append(dy)
            disp_x_list.append(dx)
        
        disp_y = np.array(disp_y_list, dtype=np.float32)
        disp_x = np.array(disp_x_list, dtype=np.float32)
        
        print("Tracking paths and performing spatiotemporal IDW interpolation...")
        interp_sst, moved_count, cf_grid, cb_grid = self.track_and_interpolate_sst(
            sst_roi_time, disp_y, disp_x,
            roi_lat_range, roi_lon_range, t_float, start_time, end_time
        )
        
        Lon, Lat = np.meshgrid(roi_lon_range, roi_lat_range)
        
        if plot_results:
            self.plot_interpolated_sst(interp_sst, Lon, Lat, roi_lon_range, roi_lat_range)
        
        print(f"\nTotal grid points moved: {moved_count}")
        
        del sst_roi_time, disp_y_list, disp_x_list, disp_y, disp_x
        gc.collect()
        
        return interp_sst, moved_count, cf_grid, cb_grid

def example_usage():
    """
    Example usage of the PSDTIM class.
    """
    # Create a PSDTIM instance
    psidim = PSDTIM(spatial_window_size=3, max_displacement=3)
    
    # Define data parameters
    nc_file_path = 'mync_interval_2.nc'
    roi_lat_range = (28, 32)
    roi_lon_range = (172, 175)
    start_time = 13
    end_time = 16
    t_float = 14.5

    # Load ROI SST data
    sst_roi_time, _, _ = psidim.load_roi_time_data(
        nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time
    )
    
    # Run the PSDTIM interpolation
    interp_sst, moved_count, cf, cb = psidim.interpolate(
        sst_roi_time, roi_lat_range, roi_lon_range,
        start_time, end_time, t_float,
        window_size=3, plot_results=True
    )
    
    # Load true SST for comparison
    file_path = 'merged_data.nc'
    true_sst = MY.get_sst_data_for_day_and_area(
        file_path, t_float * 2, roi_lat_range, roi_lon_range
    )
    
    # Plot and compute metrics
    MY.plot_Scatter_plot_graph(true_sst, interp_sst)
    MY.plot_raster_grid(interp_sst, roi_lat_range, roi_lon_range, 15, 25)
    MY.plot_raster_grid(true_sst, roi_lat_range, roi_lon_range, 15, 25)
    MY.print_ssim(interp_sst, true_sst)

    return interp_sst

if __name__ == "__main__":
    example_usage()
