import netCDF4 as nc
import numpy as np
import os
from tqdm import tqdm  # Used to display progress bars
from statsmodels.tsa.seasonal import STL


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
        sst_data (np.ndarray): SST data in degrees Celsius, shape (time, lat, lon).
        lats (np.ndarray): Latitude array for the ROI.
        lons (np.ndarray): Longitude array for the ROI.
        time_var (np.ndarray): Time variable array.
    """
    ds = nc.Dataset(nc_file_path, 'r')
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    time_var = ds.variables['time'][start_time:end_time]

    # Determine index ranges for latitude and longitude
    lat_min, lat_max = roi_lat_range
    lon_min, lon_max = roi_lon_range
    lat_min_idx = np.searchsorted(lats, lat_min)
    lat_max_idx = np.searchsorted(lats, lat_max)
    lon_min_idx = np.searchsorted(lons, lon_min)
    lon_max_idx = np.searchsorted(lons, lon_max)

    # Load SST data and slice to ROI
    sst_var = ds.variables['analysed_sst']
    sst_data = sst_var[
        start_time:end_time,
        lat_min_idx:lat_max_idx,
        lon_min_idx:lon_max_idx
    ].astype(np.float32)
    fill_value = getattr(sst_var, '_FillValue', None)

    # Replace fill values with NaN and convert to Celsius
    if fill_value is not None:
        sst_data = np.where(sst_data == fill_value, np.nan, sst_data)
    sst_data = sst_data - 273.15  # Convert Kelvin to Celsius

    ds.close()
    return sst_data, lats[lat_min_idx:lat_max_idx], lons[lon_min_idx:lon_max_idx], time_var


def save_to_nc(output_path, data, data_type, lats, lons, time_var):
    """
    Save decomposed data to a new NetCDF file.

    Args:
        output_path (str): Path for the output NetCDF file.
        data (np.ndarray): Decomposed data array, shape (time, lat, lon).
        data_type (str): Type of data (e.g., 'trend', 'seasonal', 'residual').
        lats (np.ndarray): Latitude array.
        lons (np.ndarray): Longitude array.
        time_var (np.ndarray): Time variable array.
    """
    with nc.Dataset(output_path, 'w', format='NETCDF4') as dst:
        # Create dimensions
        dst.createDimension('time', len(time_var))
        dst.createDimension('lat', len(lats))
        dst.createDimension('lon', len(lons))

        # Create variables
        time_nc = dst.createVariable('time', 'f4', ('time',))
        lat_nc  = dst.createVariable('lat',  'f4', ('lat',))
        lon_nc  = dst.createVariable('lon',  'f4', ('lon',))
        data_nc = dst.createVariable(data_type, 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)

        # Populate variables
        time_nc[:] = time_var
        lat_nc[:]  = lats
        lon_nc[:]  = lons
        data_nc[:] = data

        # Add attributes
        data_nc.units     = 'degrees Celsius'
        data_nc.long_name = f'{data_type.capitalize()} component of SST'

    print(f"{data_type.capitalize()} data saved to {output_path}")


def process_and_save_stl_components(nc_file_path, roi_lat_range, roi_lon_range,
                                    start_time, end_time, period, seasonal,
                                    trend, robust, output_dir):
    """
    Perform STL decomposition on SST data and save trend, seasonal,
    and residual components to separate NetCDF files.

    Args:
        nc_file_path (str): Path to the input NetCDF file.
        roi_lat_range (tuple): Latitude range of the ROI.
        roi_lon_range (tuple): Longitude range of the ROI.
        start_time (int): Start time-step index.
        end_time (int): End time-step index.
        period (int): Period for STL decomposition.
        seasonal (int): Seasonal window length for STL.
        trend (int): Trend window length for STL.
        robust (bool): Whether to use robust STL.
        output_dir (str): Directory to save output files.
    """
    # Load data
    sst_data, lats, lons, time_var = load_roi_time_data(
        nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time
    )

    # Initialize arrays for decomposition results
    trends        = np.full_like(sst_data, np.nan)
    seasonal_data = np.full_like(sst_data, np.nan)
    residuals     = np.full_like(sst_data, np.nan)

    # Total number of grid points for progress bar
    total_grid_points = sst_data.shape[1] * sst_data.shape[2]

    # Perform STL decomposition for each grid point
    with tqdm(total=total_grid_points, desc="STL decomposition progress", unit="grid point") as pbar:
        for lat_idx in range(sst_data.shape[1]):
            for lon_idx in range(sst_data.shape[2]):
                ts = sst_data[:, lat_idx, lon_idx]
                if np.isnan(ts).all():
                    pbar.update(1)
                    continue  # Skip if all values are NaN

                stl_result = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()
                trends[:, lat_idx, lon_idx]        = stl_result.trend
                seasonal_data[:, lat_idx, lon_idx] = stl_result.seasonal
                residuals[:, lat_idx, lon_idx]     = stl_result.resid

                pbar.update(1)

    # Save each component to NetCDF
    os.makedirs(output_dir, exist_ok=True)
    trend_path    = os.path.join(output_dir, 'trend_output_day.nc')
    seasonal_path = os.path.join(output_dir, 'seasonal_output_day.nc')
    residual_path = os.path.join(output_dir, 'residual_output_day.nc')

    save_to_nc(trend_path,    trends,        'trend',    lats, lons, time_var)
    save_to_nc(seasonal_path, seasonal_data, 'seasonal', lats, lons, time_var)
    save_to_nc(residual_path, residuals,     'residual', lats, lons, time_var)


# Example usage
if __name__ == '__main__':
    nc_file_path = r'F:\result\merged_roi.nc'  # Input NetCDF file path
    roi_lat_range = (28, 32)                   # Latitude range of interest
    roi_lon_range = (172, 175)                 # Longitude range of interest
    start_time = 0                             # Start time-step index
    end_time   = 22222                          # End time-step index (daily data)
    period     = 2222                    # Period for STL decomposition
    seasonal   = 222                          # Seasonal window length for STL
    trend      = 22                         # Trend window length (> period, odd)
    robust     = True                          # Use robust decomposition
    output_dir = r'F:\STL_result_NC'           # Directory to save outputs

    process_and_save_stl_components(
        nc_file_path, roi_lat_range, roi_lon_range,
        start_time, end_time, period,
        seasonal, trend, robust, output_dir
    )
