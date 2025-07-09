
import numpy as np
import netCDF4 as nc
from scipy.fftpack import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import os


def save_to_nc(data_array, file_path, variable_name='analysed_sst', dimensions=None):
    """
    Save an array to an NC file, supports appending data along the time dimension.

    Parameters:
    data_array: The numpy array to be saved
    file_path: NC file path
    variable_name: The variable name, default is 'data'
    dimensions: List of dimension names, if None, they will be generated automatically

    Returns:
    The length of the time dimension in the current file
    """
    # Ensure the data is a numpy array
    data_array = np.array(data_array)
    
    # Check if the file exists
    if os.path.exists(file_path):
        # The file exists, open and append data
        dataset = nc.Dataset(file_path, 'r+')
        
        # Get the current size of the time dimension
        time_dim = dataset.dimensions['time'].size
        print(f"Current time dimension size in the existing file: {time_dim}")
        
        # Append new data along the time dimension
        var = dataset.variables[variable_name]
        var[time_dim:time_dim+1, ...] = data_array
        
        # Update the time index
        dataset.variables['time'][time_dim] = time_dim
        
        # Close the file
        dataset.close()
        
        return time_dim + 1
    else:
        # The file does not exist, create a new one
        dataset = nc.Dataset(file_path, 'w')
        
        # If dimensions are not provided, generate them automatically
        if dimensions is None:
            dimensions = ['time']
            for i in range(len(data_array.shape)):
                dimensions.append(f'dim_{i}')
        
        # Create dimensions
        dataset.createDimension('time', None)  # Set time dimension as unlimited
        for i, dim_name in enumerate(dimensions[1:]):
            if i < len(data_array.shape):
                dataset.createDimension(dim_name, data_array.shape[i])
            
        # Create the time variable
        time_var = dataset.createVariable('time', 'i4', ('time',))
        time_var[0] = 0
        
        # Create the data variable
        var_dims = tuple(dimensions[:len(data_array.shape)+1])
        var = dataset.createVariable(variable_name, data_array.dtype, var_dims)
        
        # Set initial data
        var[0, ...] = data_array
        
        # Close the file
        dataset.close()
        
        print("Created a new NC file, time index starts from 0")
        return 1
    

import netCDF4 as nc

def check_nc_shape(file_path):
    """
    Check the shape and size of the NC file.
    
    Parameters:
    file_path: NC file path
    
    Returns:
    None, directly prints file information
    """
    try:
        # Open the NC file
        dataset = nc.Dataset(file_path, 'r')
        
        print(f"File: {file_path}")
        print("\nDimension Information:")
        # Print all dimensions
        for dim_name, dimension in dataset.dimensions.items():
            size = len(dimension)
            unlimited = "Unlimited" if dimension.isunlimited() else "Fixed"
            print(f"  {dim_name}: size = {size}, type = {unlimited}")
        
        print("\nVariable Information:")
        # Print all variables
        for var_name, variable in dataset.variables.items():
            dims = variable.dimensions
            shape = variable.shape
            dtype = variable.dtype
            print(f"  {var_name}:")
            print(f"    Dimensions: {dims}")
            print(f"    Shape: {shape}")
            print(f"    Data type: {dtype}")
            
            # If it's a time variable, display the time index
            if var_name == 'time':
                time_indices = variable[:]
                print(f"    Time index: {time_indices}")
        
        # Close the file
        dataset.close()
        
    except Exception as e:
        print(f"Error reading the file: {str(e)}")



# # Create an example array
# data = np.random.rand(10, 10)

# # First call, create a new file
# time_dim = save_to_nc(data, 'output.nc')
# print(f"Current time dimension size: {time_dim}")

# # Second call, append data
# data2 = np.random.rand(10, 10)
# time_dim = save_to_nc(data2, 'output.nc')
# print(f"Current time dimension size: {time_dim}")


# Load SST data for a specified ROI and time range
def load_roi_time_data(nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time):
    """
    Load SST data for a specified ROI and time range from a NetCDF file.
    
    Parameters:
    - nc_file_path: NetCDF file path
    - roi_lat_range: Latitude range of ROI (min_lat, max_lat)
    - roi_lon_range: Longitude range of ROI (min_lon, max_lon)
    - start_time: Start time step index (starting from 0)
    - end_time: End time step index (starting from 0, inclusive)
    
    Returns:
    - sst_roi_time: SST data within the ROI and time range (time, lat, lon)
    - lats_roi: Latitude array for the ROI
    - lons_roi: Longitude array for the ROI
    """
    ds = nc.Dataset(nc_file_path, 'r')
    
    # Extract latitude and longitude data
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    
    # Find the indices for the ROI
    lat_min, lat_max = roi_lat_range
    lon_min, lon_max = roi_lon_range
    lat_min_idx = np.searchsorted(lats, lat_min)
    lat_max_idx = np.searchsorted(lats, lat_max)
    lon_min_idx = np.searchsorted(lons, lon_min)
    lon_max_idx = np.searchsorted(lons, lon_max)
    
    # Ensure the time indices are valid
    num_times = len(ds.dimensions['time'])
    if start_time < 0 or end_time >= num_times:
        raise IndexError("start_time or end_time is out of bounds.")
    
    # Load the SST data within the ROI and time range
    sst_var = ds.variables['analysed_sst']
    sst_roi_time = sst_var[start_time:end_time + 1, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx].astype(np.float32)
    
    # Handle missing values
    fill_value = getattr(sst_var, '_FillValue', None)
    if fill_value is not None:
        sst_roi_time = np.where(sst_roi_time == fill_value, np.nan, sst_roi_time)
    else:
        sst_roi_time = np.where(np.isnan(sst_roi_time), np.nan, sst_roi_time)
    
    sst_roi_time = sst_roi_time - 273.15  # Convert to Celsius
    lats_roi = lats[lat_min_idx:lat_max_idx]
    lons_roi = lons[lon_min_idx:lon_max_idx]
    
    ds.close()
    return sst_roi_time, lats_roi, lons_roi




# Calculate and print SSIM
def print_ssim(array1, array2):
    # Calculate SSIM and handle potential issues with data range
    ssim_value, _ = ssim(array1, array2, data_range=array2.max() - array2.min(), full=True)
    print(f"SSIM: {ssim_value:.3f}")









# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(true_values, predicted_values):
    """
    Calculate various regression evaluation metrics.

    Parameters:
    - true_values (array-like): True values
    - predicted_values (array-like): Predicted values

    Returns:
    - mse (float): Mean squared error
    - rmse (float): Root mean squared error
    - mae (float): Mean absolute error
    - mre (float): Mean relative error
    - mape (float): Mean absolute percentage error
    - ssim_value (float): Structural similarity index
    - r2 (float): R-squared value
    """
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    mre = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

    # Calculate SSIM, specifying data_range
    ssim_value, _ = ssim(true_values, predicted_values, data_range=predicted_values.max() - predicted_values.min(), full=True)

    # Calculate R^2
    r2 = r2_score(true_values, predicted_values)

    return mse, rmse, mae, mre, mape, ssim_value, r2


def plot_Scatter_plot_graph(true_values, predicted_values):
    """
    Plot scatter plot, regression line, 1:1 line, add marginal histograms and density area plot, and display regression metrics.

    Parameters:
    - true_values (array-like): True values
    - predicted_values (array-like): Predicted values
    """
    # Flatten arrays
    true_values_flat = true_values.flatten()
    predicted_values_flat = predicted_values.flatten()

    # Linear regression model fit
    model = LinearRegression()
    true_values_reshaped = true_values_flat.reshape(-1, 1)
    model.fit(true_values_reshaped, predicted_values_flat)
    fit_line = model.predict(true_values_reshaped)
    slope = model.coef_[0]

    # Calculate regression metrics
    mse, rmse, mae, mre, mape, ssim_value, r2 = calculate_metrics(true_values, predicted_values)
    n_points = len(true_values_flat)

    # Create DataFrame for JointGrid
    data = pd.DataFrame({'True': true_values_flat, 'Predicted': predicted_values_flat})

    # Set plot style
    sns.set(style="white", color_codes=True)

    # Create JointGrid object
    g = sns.JointGrid(data=data, x="True", y="Predicted", space=0, height=8)

    # Plot main chart: scatter plot and regression line
    g = g.plot_joint(
        sns.regplot,
        scatter_kws={"s": 30, "color": "#6A8FA5", "alpha": 0.5, "label": "Data Points"},
        line_kws={"color": "blue", "alpha": 0.8, "label": "Fit Line"}
    )

    # Plot 1:1 line, starting from the minimum value of the data range
    min_val = min(true_values_flat.min(), predicted_values_flat.min())  # Start from the minimum value of the data
    max_val = max(true_values_flat.max(), predicted_values_flat.max())
    
    # Adjust axis range to multiples of 0.5
    min_val = np.floor(min_val * 2) / 2  # Round down to the nearest 0.5 multiple
    max_val = np.ceil(max_val * 2) / 2  # Round up to the nearest 0.5 multiple
    
    # Plot 1:1 line
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

    # Set the same axis range and aspect ratio
    g.ax_joint.set_xlim(min_val, max_val)
    g.ax_joint.set_ylim(min_val, max_val)
    g.ax_joint.set_aspect('equal', adjustable='box')

    # Set axis tick intervals to 0.5
    g.ax_joint.set_xticks(np.arange(min_val, max_val + 0.5, 0.5))
    g.ax_joint.set_yticks(np.arange(min_val, max_val + 0.5, 0.5))

    # Plot the marginal histogram on top
    g.ax_marg_x.hist(
        data["True"],
        bins=25,
        color="#66c2a5",
        edgecolor="black",
        alpha=0.6,  # Set transparency for the histogram
        density=True  # Normalize the histogram to overlay with density plot
    )

    # Overlay the density plot on top of the marginal histogram
    sns.kdeplot(
        data=data,
        x="True",
        ax=g.ax_marg_x,
        color="#1f78b4",
        alpha=0.5,  # Set transparency for the density plot
        linewidth=2
    )

    # Plot the marginal histogram on the right side
    g.ax_marg_y.hist(
        data["Predicted"],
        bins=25,
        orientation="horizontal",
        color="#fc8d62",
        edgecolor="black",
        alpha=0.6,  # Set transparency for the histogram
        density=True  # Normalize the histogram to overlay with density plot
    )

    # Overlay the density plot on the right marginal histogram
    sns.kdeplot(
        data=data,
        y="Predicted",
        ax=g.ax_marg_y,
        color="#e31a1c",
        alpha=0.5,  # Set transparency for the density plot
        linewidth=2
    )

    # Add text labels (regression equation and R^2 value)
    metrics_text = (f'N = {n_points}\n'
                    f'MSE = {mse:.3f}\n'
                    f'RMSE = {rmse:.3f}\n'
                    f'MAE = {mae:.3f}\n'
                    f'MRE = {mre:.2f}%\n'
                    f'MAPE = {mape:.2f}%\n'
                    f'SSIM = {ssim_value:.3f}\n'
                    f'R² = {r2:.3f}\n'
                    f'Slope = {slope:.3f}')

    # Adjust text position to fit the data range
    g.ax_joint.text(
        0.05, 0.95,
        metrics_text,
        transform=g.ax_joint.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.6)
    )

    # Set axis labels
    g.set_axis_labels('True Values °C', 'Predicted Values °C')

    # Add title
    plt.subplots_adjust(top=0.95)

    # Add legend
    handles, labels = g.ax_joint.get_legend_handles_labels()
    # Remove duplicate labels (because sns.regplot and manually drawn 1:1 line might overlap)
    unique_labels = dict(zip(labels, handles))
    g.ax_joint.legend(unique_labels.values(), unique_labels.keys(), loc='lower right')

    # Show grid
    g.ax_joint.grid(True)

    # Display the plot
    plt.show()





# Get the specified number of days' grid data for a given area

# Get the grid data for a specified time index
def get_sst_data_for_day_and_area(file_path, day, lat_range, lon_range):
    # Open NC file
    dataset = nc.Dataset(file_path)

    # Extract latitude and longitude data
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]
    
    # Find the indices for the specified longitude and latitude range
    lon_min_idx = np.searchsorted(lons, lon_range[0])
    lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
    lat_min_idx = np.searchsorted(lats, lat_range[0])
    lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

    # Get sea surface temperature data for the specified range and day
    sst_data = dataset.variables['analysed_sst'][day, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
    
    # Handle invalid values
    if hasattr(dataset.variables['analysed_sst'], '_FillValue'):
        fill_value = dataset.variables['analysed_sst']._FillValue
        sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data).filled(np.nan)  # Fill invalid values with np.nan
    else:
        sst_data_masked = np.ma.masked_invalid(sst_data).filled(np.nan)

    # Close file
    dataset.close()
    
    # Convert temperature from Kelvin to Celsius
    sst_data_masked = sst_data_masked - 273.15

    return sst_data_masked


# Load and prepare grid data from the entire spatial range of an NC file
def load_and_prepare_data(nc_file_path):
    """Load and prepare SST data from a NetCDF file."""
    ds = nc.Dataset(nc_file_path)
    sst = ds.variables['sst'][:]  # Assuming the dimensions are (time, lat, lon)
    sst = np.nan_to_num(sst, nan=np.nan)
    # Check and handle anomalies
    sst = np.clip(sst, -1e10, 1e10)  # Limit values to a reasonable range
    return sst



import netCDF4 as nc
import numpy as np

# Get the grid data for a specified time range and specified latitude and longitude range from the NC file,
# fetching data from specific time indices, not corresponding to the time index in the file
def get_sst_data_for_time_range_and_area(file_path, start_day, end_day, lat_range, lon_range):
    # Open NC file
    dataset = nc.Dataset(file_path)

    # Extract latitude and longitude data
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]

    # Find the indices for the specified longitude and latitude range
    lon_min_idx = np.searchsorted(lons, lon_range[0])
    lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
    lat_min_idx = np.searchsorted(lats, lat_range[0])
    lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

    # Extract sea surface temperature data for the specified time range
    sst_data_list = []
    for day in range(start_day, end_day + 1):
        sst_data = dataset.variables['analysed_sst'][day, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
        
        # Handle invalid values
        if hasattr(dataset.variables['analysed_sst'], '_FillValue'):
            fill_value = dataset.variables['analysed_sst']._FillValue
            sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data).filled(np.nan)  # Fill invalid values with np.nan
        else:
            sst_data_masked = np.ma.masked_invalid(sst_data).filled(np.nan)

        # Convert temperature from Kelvin to Celsius
        sst_data_masked = sst_data_masked - 273.15

        sst_data_list.append(sst_data_masked)

    # Close file
    dataset.close()

    return np.array(sst_data_list)



# Open NC file and visualize the sea surface temperature (SST) grid for the specified time range and specified latitude and longitude range
def plot_sst_for_days(file_path, days, lat_range, lon_range):
    # Open NC file
    dataset = nc.Dataset(file_path)

    # Extract latitude and longitude data
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]
    
    # Find the indices for the specified longitude and latitude range
    lon_min_idx = np.searchsorted(lons, lon_range[0])
    lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
    lat_min_idx = np.searchsorted(lats, lat_range[0])
    lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

    # Loop through the specified days
    for day in days:
        # Get sea surface temperature data for the specified range and day
        sst_data = dataset.variables['analysed_sst'][day, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
        
        # Handle invalid values
        if hasattr(dataset.variables['analysed_sst'], '_FillValue'):
            fill_value = dataset.variables['analysed_sst']._FillValue
            sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data)-273.15
        else:
            sst_data_masked = np.ma.masked_invalid(sst_data)-273.15

        # Prepare for plotting
        plt.figure(figsize=(12, 6))

        # Plot the data using imshow
        plt.imshow(sst_data_masked, origin='lower', cmap='jet', 
                   extent=[lons[lon_min_idx], lons[lon_max_idx-1], lats[lat_min_idx], lats[lat_max_idx-1]])

        # Add color bar
        plt.colorbar(label='Sea Surface Temperature (°C)')

        # Add title and axis labels
        plt.title(f'Sea Surface Temperature (SST) for day {day+1}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()



# Function to print grid indices for a given time step and area
def print_grid_indices_at_time(nc_file_path, day_index, lat_range, lon_range):
    # Open NC file
    dataset = nc.Dataset(nc_file_path)

    # Extract latitude and longitude data
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]

    # Find the indices for the specified longitude and latitude range
    lon_min_idx = np.searchsorted(lons, lon_range[0])
    lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
    lat_min_idx = np.searchsorted(lats, lat_range[0])
    lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

    # Print results
    print(f"At time index {day_index}, grid point indices for the latitude range {lat_range} and longitude range {lon_range}:")
    print(f"Longitude indices: {lon_min_idx} to {lon_max_idx - 1}")
    print(f"Latitude indices: {lat_min_idx} to {lat_max_idx - 1}")

    # Optionally return these indices if further processing or testing is needed
    return (lon_min_idx, lon_max_idx - 1), (lat_min_idx, lat_max_idx - 1)

# Example usage
# nc_file_path = 'sst.day.mean.2006.nc'
# day_index = 10
# lat_range = (28, 29)  # Example latitude range
# lon_range = (170, 171)  # Example longitude range
# print_grid_indices_at_time(nc_file_path, day_index, lat_range, lon_range)



# Basic version to calculate scatter plot and compute metrics
def calculate_error_metrics(true_values, predicted_values):
    """
    Calculate and print MSE, RMSE, and MAE.
    """
    # Ensure the shapes of the true values and predicted values are consistent, and remove NaN values
    mask = ~np.isnan(true_values) & ~np.isnan(predicted_values)
    true_values = true_values[mask]
    predicted_values = predicted_values[mask]
    
    # Calculate MSE
    mse = np.mean((true_values - predicted_values) ** 2)
    # Calculate RMSE
    rmse = np.sqrt(mse)
    # Calculate MAE
    mae = np.mean(np.abs(true_values - predicted_values))
    
    print(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")


def plot_consistency_graph(true_values, predicted_values):
    """
    Plot a consistency graph of predicted values vs true values.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--')  # Plot 1:1 line
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Consistency Graph')
    plt.grid(True)
    plt.show()


# Input NC file path and print variable information from the NC file
def print_myNC_information(file):
    # Open NC file
    dataset = nc.Dataset(file)

    # Get all variable keys
    all_vars = dataset.variables.keys()
    print("Variables in the dataset:", all_vars)

    # Get information about all variables
    all_vars_info = list(dataset.variables.items())
    print("\nVariables information:")
    for var in all_vars_info:
       print(var[0], var[1])

# Example usage
# file = '20060129-REMSS-L4HRfnd-GLOB-v01-fv01-mw_ir_OI.nc'
# print_myNC_information(file)


# Input NC file and plot the sea surface temperature (SST) for a specific date and region
def plot_sst_for_date_and_region(nc_file, date_index, lon_range, lat_range):
    """
    Plot Sea Surface Temperature (SST) for a specific date and region in a NetCDF file.

    Parameters:
    - nc_file: str, Path to the NetCDF file.
    - date_index: int, Index of the date in the dataset.
    - lon_range: tuple, Longitude range (min_lon, max_lon).
    - lat_range: tuple, Latitude range (min_lat, max_lat).
    """
    # Open the NetCDF file
    dataset = nc.Dataset(nc_file)

    # Extract latitude and longitude data
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]

    # Find the indices for the specified longitude and latitude range
    lon_min_idx = np.where(lons >= lon_range[0])[0][0]
    lon_max_idx = np.where(lons <= lon_range[1])[0][-1]
    lat_min_idx = np.where(lats >= lat_range[0])[0][0]
    lat_max_idx = np.where(lats <= lat_range[1])[0][-1]

    # Get the sea surface temperature (SST) data for the specified date and range
    sst_data = dataset.variables['analysed_sst'][date_index, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]

    # Check if _FillValue attribute exists and mask invalid data
    if hasattr(dataset.variables['analysed_sst'], '_FillValue'):
        fill_value = dataset.variables['analysed_sst']._FillValue
        sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data)
    else:
        sst_data_masked = np.ma.masked_invalid(sst_data)

    sst_data_masked = sst_data_masked - 273.15  # Convert from Kelvin to Celsius
    # Prepare for plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(sst_data_masked, origin='lower', cmap='jet', 
               extent=[lons[lon_min_idx], lons[lon_max_idx], lats[lat_min_idx], lats[lat_max_idx]])
    plt.colorbar(label='Sea Surface Temperature (°C)')
    plt.title(f'SST for day {date_index} in the specified region')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Example usage:
# file_path = '20060129-REMSS-L4HRfnd-GLOB-v01-fv01-mw_ir_OI.nc'
# plot_sst_for_date_and_region(file_path, 0, (170, 175), (28, 32))



def print_grid_shape(interpolated_grid):
    """
    Print the size (rows and columns) of an interpolated grid image.

    Parameters:
    interpolated_grid -- The interpolated grid image, a NumPy array.
    """
    rows, cols = interpolated_grid.shape
    print(f"The grid image has {rows} rows and {cols} columns.")


import numpy as np
import matplotlib.pyplot as plt


from matplotlib.colors import Normalize
def plot_raster_grid(raster_data, lat_range, lon_range, vmin=None, vmax=None, title=None):
    """
    Function to plot raster data.

    Parameters:
    raster_data : np.array
        The raster data array.
    lat_range : tuple
        Latitude range in the format (min_lat, max_lat).
    lon_range : tuple
        Longitude range in the format (min_lon, max_lon).
    vmin : float, optional
        Minimum value for the color bar.
    vmax : float, optional
        Maximum value for the color bar.
    title : str, optional
        Image title.
    """
    # Create the image
    plt.figure(figsize=(12, 6))

    # Plot the raster grid
    plt.imshow(
        raster_data,
        origin='lower',
        cmap='jet',
        extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
        norm=Normalize(vmin=vmin, vmax=vmax)
    )

    # Add color bar
    plt.colorbar(label='Value')

    # Set title and axis labels
    if title is not None:
        plt.title(title)
    else:
        plt.title('Raster Grid Visualization')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°N)')
    
    # Show the image
    plt.show()


def print_grid_point_value(images, time_index, lat_idx, lon_idx):
    """
    Print the value at the specified time step and grid point.

    Parameters:
    images : np.array
        The array containing SST data, with dimensions (time, lat, lon).
    time_index : int
        The index for the time step.
    lat_idx : int
        The index for the latitude.
    lon_idx : int
        The index for the longitude.
    """
    if time_index < 0 or time_index >= images.shape[0]:
        print(f"Invalid time index: {time_index}")
        return
    if lat_idx < 0 or lat_idx >= images.shape[1]:
        print(f"Invalid latitude index: {lat_idx}")
        return
    if lon_idx < 0 or lon_idx >= images.shape[2]:
        print(f"Invalid longitude index: {lon_idx}")
        return

    value = images[time_index, lat_idx, lon_idx]
    print(f"Value at time {time_index}, grid point ({lat_idx}, {lon_idx}): {value}")

# Example usage
# nc_file_path = 'sst.day.mean.2006.nc'
# images = load_and_prepare_data(nc_file_path)
# print_grid_shape(images[0])

# Print value at a specific time step and grid point
# time_index = 4
# lat_idx = 112
# lon_idx = 680
# print_grid_point_value(images, time_index, lat_idx, lon_idx)

import netCDF4 as nc
import numpy as np

# Output the structure of the NC file
def check_nc_file_structure(nc_file_path):
    """Check the structure of the NetCDF file"""
    ds = nc.Dataset(nc_file_path)
    print(ds)
    print(ds.variables.keys())
    for var in ds.variables:
        print(f"{var}: {ds.variables[var].dimensions}, {ds.variables[var].shape}")

# Example usage
# nc_file_path = 'sst.day.mean.2006.nc'
# check_nc_file_structure(nc_file_path)



import netCDF4 as nc
import numpy as np

def get_lat_lon_indices(ds, lat_range, lon_range):
    """Convert latitude and longitude range into index range"""
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    min_lat_idx = np.abs(lat - lat_range[0]).argmin()
    max_lat_idx = np.abs(lat - lat_range[1]).argmin()
    min_lon_idx = np.abs(lon - lon_range[0]).argmin()
    max_lon_idx = np.abs(lon - lon_range[1]).argmin()
    return (min_lat_idx, max_lat_idx), (min_lon_idx, max_lon_idx)

def print_nc_grid_range(nc_file_path, time_index, lat_range, lon_range):
    """
    Print SST values in the specified time step and latitude/longitude range of the NetCDF file.

    Parameters:
    nc_file_path : str
        Path to the NetCDF file.
    time_index : int
        Index of the time step.
    lat_range : tuple
        Latitude range in the format (min_lat, max_lat).
    lon_range : tuple
        Longitude range in the format (min_lon, max_lon).
    """
    ds = nc.Dataset(nc_file_path)
    sst = ds.variables['sst'][:]  # Assuming SST data variable name is 'sst'

    # Convert masked array to NaN
    if np.ma.is_masked(sst):
        sst = sst.filled(np.nan)

    if time_index < 0 or time_index >= sst.shape[0]:
        print(f"Invalid time index: {time_index}")
        return

    (min_lat_idx, max_lat_idx), (min_lon_idx, max_lon_idx) = get_lat_lon_indices(ds, lat_range, lon_range)

    if min_lat_idx < 0 or max_lat_idx >= sst.shape[1] or min_lat_idx > max_lat_idx:
        print(f"Invalid latitude range: {lat_range}")
        return
    if min_lon_idx < 0 or max_lon_idx >= sst.shape[2] or min_lon_idx > max_lon_idx:
        print(f"Invalid longitude range: {lon_range}")
        return

    sst_values = sst[time_index, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
    print(f"SST values at time {time_index}, latitude range {lat_range}, longitude range {lon_range}:\n{sst_values}")

# Example usage
# nc_file_path = 'sst.day.mean.2006.nc'

# Print the grid values for the specified time step and latitude/longitude range
# time_index = 4
# lat_range = (28, 29)  # Example latitude range
# lon_range = (170, 171)  # Example longitude range
# print_nc_grid_range(nc_file_path, time_index, lat_range, lon_range)



# Visualize the displacement field
def plot_displacement_field(displacement_field):
    """Plot the displacement field as a vector field"""
    plt.figure(figsize=(10, 10))
    height, width, _ = displacement_field.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    U = displacement_field[:, :, 1]
    V = displacement_field[:, :, 0]
    plt.quiver(X, Y, U, V, angles='xy')
    plt.gca().invert_yaxis()
    plt.title("Displacement Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

import netCDF4 as nc
from netCDF4 import num2date

def print_nc_file_time_indices(nc_file_path):
    """
    Print the time indices and their corresponding actual date-time in a NetCDF file.
    
    Parameters:
    - nc_file_path: Path to the NC file.
    """
    with nc.Dataset(nc_file_path, 'r') as ds:
        # Get the time variable
        time_var = ds.variables['time']
        time_units = time_var.units
        time_calendar = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'
        
        # Convert time indices to actual date-time
        time_indices = time_var[:]
        dates = num2date(time_indices, units=time_units, calendar=time_calendar)
        
        # Print time indices and their corresponding actual date-time
        for idx, date in zip(time_indices, dates):
            print(f"Index: {idx}, Date: {date}")

# Example usage:
  # Replace with your NC file path
# print_nc_file_time_indices(nc_file_path)


# Print SST values at a specific grid point and time in the NC file
def print_sst_at_point_and_time(nc_file_path, time_index, lat_target, lon_target):
    """
    Print the sea surface temperature (SST) value at a specified time step and grid point in the NC file.

    Parameters:
    - nc_file_path: Path to the NC file.
    - time_index: Index of the specified time step.
    - lat_target: Latitude of the target grid point.
    - lon_target: Longitude of the target grid point.
    """
    # Open the NC file
    dataset = nc.Dataset(nc_file_path)

    # Get variables
    sst = dataset.variables['sst']  # Assuming the variable name is 'sst'
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]

    # Find the indices of the target latitude and longitude
    lat_idx = (np.abs(lats - lat_target)).argmin()
    lon_idx = (np.abs(lons - lon_target)).argmin()

    # Get and print the SST value at the specified time and grid point
    sst_value = sst[time_index, lat_idx, lon_idx]
    print(f"SST at time index {time_index}, lat {lat_target}, lon {lon_target}: {sst_value}°C")
