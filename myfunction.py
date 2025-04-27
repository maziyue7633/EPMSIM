
import numpy as np
import netCDF4 as nc
from scipy.fftpack import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import os


def save_to_nc(data_array, file_path, variable_name='analysed_sst', dimensions=None):
    """
    将数组保存到NC文件，支持在时间维度上追加数据
    
    参数:
    data_array: 要保存的numpy数组
    file_path: NC文件路径
    variable_name: 变量名称，默认为'data'
    dimensions: 维度名称列表，如果为None则自动生成
    
    返回:
    当前文件的时间维度长度
    """
    # 确保数据是numpy数组
    data_array = np.array(data_array)
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 文件已存在，打开并追加数据
        dataset = nc.Dataset(file_path, 'r+')
        
        # 获取当前时间维度大小
        time_dim = dataset.dimensions['time'].size
        print(f"现有文件的时间维度大小: {time_dim}")
        
        # 在时间维度上添加新数据
        var = dataset.variables[variable_name]
        var[time_dim:time_dim+1, ...] = data_array
        
        # 更新时间索引
        dataset.variables['time'][time_dim] = time_dim
        
        # 关闭文件
        dataset.close()
        
        return time_dim + 1
    else:
        # 文件不存在，创建新文件
        dataset = nc.Dataset(file_path, 'w')
        
        # 如果未提供维度名称，则自动生成
        if dimensions is None:
            dimensions = ['time']
            for i in range(len(data_array.shape)):
                dimensions.append(f'dim_{i}')
        
        # 创建维度
        dataset.createDimension('time', None)  # 时间维度设为无限制
        for i, dim_name in enumerate(dimensions[1:]):
            if i < len(data_array.shape):
                dataset.createDimension(dim_name, data_array.shape[i])
            
        # 创建时间变量
        time_var = dataset.createVariable('time', 'i4', ('time',))
        time_var[0] = 0
        
        # 创建数据变量
        var_dims = tuple(dimensions[:len(data_array.shape)+1])
        var = dataset.createVariable(variable_name, data_array.dtype, var_dims)
        
        # 设置初始数据
        var[0, ...] = data_array
        
        # 关闭文件
        dataset.close()
        
        print("创建了新的NC文件，时间索引从0开始")
        return 1
    

import netCDF4 as nc

def check_nc_shape(file_path):
    """
    查看NC文件的形状大小
    
    参数:
    file_path: NC文件路径
    
    返回:
    无，直接打印文件信息
    """
    try:
        # 打开NC文件
        dataset = nc.Dataset(file_path, 'r')
        
        print(f"文件: {file_path}")
        print("\n维度信息:")
        # 打印所有维度
        for dim_name, dimension in dataset.dimensions.items():
            size = len(dimension)
            unlimited = "无限制" if dimension.isunlimited() else "固定"
            print(f"  {dim_name}: 大小 = {size}, 类型 = {unlimited}")
        
        print("\n变量信息:")
        # 打印所有变量
        for var_name, variable in dataset.variables.items():
            dims = variable.dimensions
            shape = variable.shape
            dtype = variable.dtype
            print(f"  {var_name}:")
            print(f"    维度: {dims}")
            print(f"    形状: {shape}")
            print(f"    数据类型: {dtype}")
            
            # 如果是时间变量，显示时间索引
            if var_name == 'time':
                time_indices = variable[:]
                print(f"    时间索引: {time_indices}")
        
        # 关闭文件
        dataset.close()
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")



# # 创建一个示例数组
# data = np.random.rand(10, 10)

# # 第一次调用，创建新文件
# time_dim = save_to_nc(data, 'output.nc')
# print(f"当前时间维度大小: {time_dim}")

# # 再次调用，追加数据
# data2 = np.random.rand(10, 10)
# time_dim = save_to_nc(data2, 'output.nc')
# print(f"当前时间维度大小: {time_dim}")


# 加载指定ROI和时间范围的SST数据
def load_roi_time_data(nc_file_path, roi_lat_range, roi_lon_range, start_time, end_time):
    """
    加载NetCDF文件中指定ROI和时间范围的SST数据。
    
    参数：
    - nc_file_path: NetCDF文件路径
    - roi_lat_range: ROI的纬度范围 (min_lat, max_lat)
    - roi_lon_range: ROI的经度范围 (min_lon, max_lon)
    - start_time: 起始时间步索引（从0开始）
    - end_time: 结束时间步索引（从0开始，包含）
    
    返回：
    - sst_roi_time: ROI和时间范围内的SST数据 (time, lat, lon)
    - lats_roi: ROI的纬度数组
    - lons_roi: ROI的经度数组
    """
    ds = nc.Dataset(nc_file_path, 'r')
    
    # 提取经纬度数据
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    
    # 找到ROI的索引
    lat_min, lat_max = roi_lat_range
    lon_min, lon_max = roi_lon_range
    lat_min_idx = np.searchsorted(lats, lat_min)
    lat_max_idx = np.searchsorted(lats, lat_max)
    lon_min_idx = np.searchsorted(lons, lon_min)
    lon_max_idx = np.searchsorted(lons, lon_max)
    
    # 确保时间步索引有效
    num_times = len(ds.dimensions['time'])
    if start_time < 0 or end_time >= num_times:
        raise IndexError("start_time 或 end_time 超出时间步范围。")
    
    # 加载ROI和时间范围内的SST数据
    sst_var = ds.variables['analysed_sst']
    sst_roi_time = sst_var[start_time:end_time + 1, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx].astype(np.float32)
    
    # 处理缺失值
    fill_value = getattr(sst_var, '_FillValue', None)
    if fill_value is not None:
        sst_roi_time = np.where(sst_roi_time == fill_value, np.nan, sst_roi_time)
    else:
        sst_roi_time = np.where(np.isnan(sst_roi_time), np.nan, sst_roi_time)
    
    sst_roi_time = sst_roi_time - 273.15  # 转换为摄氏度
    lats_roi = lats[lat_min_idx:lat_max_idx]
    lons_roi = lons[lon_min_idx:lon_max_idx]
    
    ds.close()
    return sst_roi_time, lats_roi, lons_roi




# 计算并且打印出来SSIM
def print_ssim(array1, array2):
    # Calculate SSIM and handle potential issues with data range
    ssim_value, _ = ssim(array1, array2, data_range=array2.max() - array2.min(), full=True)
    print(f"SSIM: {ssim_value:.3f}")



# 从TXT当中读取插值栅格数据的函数存为2维数组
def read_2d_raster(file_path):
    """
    读取二维插值栅格数据
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = []
    for line in lines:
        line = line.strip()
        if line:  # 确保不是空行
            matrix.append([float(x) for x in line.split(',')])

    return np.array(matrix)


# 从TXT当中读取插值栅格数据的函数存为3维数组
def read_space_predicted_temps(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrices = []
    current_matrix = []
    for line in lines:
        line = line.strip()
        if line.startswith("Day"):
            if current_matrix:
                matrices.append(np.array(current_matrix))
                current_matrix = []
        elif line:
            current_matrix.append([float(x) for x in line.split(',')])

    if current_matrix:
        matrices.append(np.array(current_matrix))

    return np.array(matrices)


# ////////////////////////////从TXT当中读取数据存为3维数组的函数///////////////////////////
def read_predicted_temps(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    matrices = []
    current_matrix = []
    for line in lines:
        line = line.strip()
        if line.startswith("Day") or line.startswith("Array"):
            if current_matrix:
                matrices.append(np.array(current_matrix))
                current_matrix = []
        elif line:
            current_matrix.append([float(x) for x in line.split(',')])
    
    if current_matrix:
        matrices.append(np.array(current_matrix))
    
    return np.array(matrices)







# 导入必要的库
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
    计算各种回归评估指标。

    参数:
    - true_values (array-like): 真实值
    - predicted_values (array-like): 预测值

    返回:
    - mse (float): 均方误差
    - rmse (float): 均方根误差
    - mae (float): 平均绝对误差
    - mre (float): 平均相对误差
    - mape (float): 平均绝对百分比误差
    - ssim_value (float): 结构相似性指数
    - r2 (float): R方值
    """
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    mre = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

    # 计算 SSIM，指定 data_range
    ssim_value, _ = ssim(true_values, predicted_values, data_range=predicted_values.max() - predicted_values.min(), full=True)

    # 计算 R^2
    r2 = r2_score(true_values, predicted_values)

    return mse, rmse, mae, mre, mape, ssim_value, r2


def plot_Scatter_plot_graph(true_values, predicted_values):
    """
    绘制散点图、回归线、1:1线，添加边际直方图和密度面积图，并显示回归指标。

    参数:
    - true_values (array-like): 真实值
    - predicted_values (array-like): 预测值
    """
    # 展平数组
    true_values_flat = true_values.flatten()
    predicted_values_flat = predicted_values.flatten()

    # 线性回归模型拟合
    model = LinearRegression()
    true_values_reshaped = true_values_flat.reshape(-1, 1)
    model.fit(true_values_reshaped, predicted_values_flat)
    fit_line = model.predict(true_values_reshaped)
    slope = model.coef_[0]

    # 计算回归指标
    mse, rmse, mae, mre, mape, ssim_value, r2 = calculate_metrics(true_values, predicted_values)
    n_points = len(true_values_flat)

    # 创建 DataFrame 用于 JointGrid
    data = pd.DataFrame({'True': true_values_flat, 'Predicted': predicted_values_flat})

    # 设置图形风格
    sns.set(style="white", color_codes=True)

    # 创建 JointGrid 对象
    g = sns.JointGrid(data=data, x="True", y="Predicted", space=0, height=8)

    # 绘制主图：散点图和回归线
    g = g.plot_joint(
        sns.regplot,
        scatter_kws={"s": 30, "color": "#6A8FA5", "alpha": 0.5, "label": "Data Points"},
        line_kws={"color": "blue", "alpha": 0.8, "label": "Fit Line"}
    )

    # 绘制1:1线，起点从数据范围的最小值开始
    min_val = min(true_values_flat.min(), predicted_values_flat.min())  # 从数据最小值开始
    max_val = max(true_values_flat.max(), predicted_values_flat.max())
    
    # 调整坐标轴范围为0.5的倍数
    min_val = np.floor(min_val * 2) / 2  # 向下取整到0.5的倍数
    max_val = np.ceil(max_val * 2) / 2  # 向上取整到0.5的倍数
    
    # 绘制1:1线
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

    # 设置相同的坐标轴范围和比例
    g.ax_joint.set_xlim(min_val, max_val)
    g.ax_joint.set_ylim(min_val, max_val)
    g.ax_joint.set_aspect('equal', adjustable='box')

    # 设置坐标轴的刻度间隔为 0.5
    g.ax_joint.set_xticks(np.arange(min_val, max_val + 0.5, 0.5))
    g.ax_joint.set_yticks(np.arange(min_val, max_val + 0.5, 0.5))

    # 绘制上方的边际直方图
    g.ax_marg_x.hist(
        data["True"],
        bins=25,
        color="#66c2a5",
        edgecolor="black",
        alpha=0.6,  # 设置直方图的透明度
        density=True  # 使直方图归一化，以便与密度图叠加
    )

    # 在上方边际图上叠加密度图
    sns.kdeplot(
        data=data,
        x="True",
        ax=g.ax_marg_x,
        color="#1f78b4",
        alpha=0.5,  # 设置密度图的透明度
        linewidth=2
    )

    # 绘制右侧的边际直方图
    g.ax_marg_y.hist(
        data["Predicted"],
        bins=25,
        orientation="horizontal",
        color="#fc8d62",
        edgecolor="black",
        alpha=0.6,  # 设置直方图的透明度
        density=True  # 使直方图归一化，以便与密度图叠加
    )

    # 在右侧边际图上叠加密度图
    sns.kdeplot(
        data=data,
        y="Predicted",
        ax=g.ax_marg_y,
        color="#e31a1c",
        alpha=0.5,  # 设置密度图的透明度
        linewidth=2
    )

    # 添加文本标签（回归方程和R^2值）
    metrics_text = (f'N = {n_points}\n'
                    f'MSE = {mse:.3f}\n'
                    f'RMSE = {rmse:.3f}\n'
                    f'MAE = {mae:.3f}\n'
                    f'MRE = {mre:.2f}%\n'
                    f'MAPE = {mape:.2f}%\n'
                    f'SSIM = {ssim_value:.3f}\n'
                    f'R² = {r2:.3f}\n'
                    f'Slope = {slope:.3f}')

    # 调整文本位置以适应数据范围
    g.ax_joint.text(
        0.05, 0.95,
        metrics_text,
        transform=g.ax_joint.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.6)
    )

    # 设置轴标签
    g.set_axis_labels('True Values °C', 'Predicted Values °C')

    # 添加标题
    plt.subplots_adjust(top=0.95)

    # 添加图例
    handles, labels = g.ax_joint.get_legend_handles_labels()
    # 移除重复标签（因为sns.regplot和手动绘制的1:1线可能会重复）
    unique_labels = dict(zip(labels, handles))
    g.ax_joint.legend(unique_labels.values(), unique_labels.keys(), loc='lower right')

    # 显示网格
    g.ax_joint.grid(True)

    # 显示图形
    plt.show()





# # 示例使用
# if __name__ == "__main__":
#     # 创建模拟数据
#     np.random.seed(10)
#     x = np.random.normal(5, 1, 100)
#     y = 0.41 * x + 4.27 + np.random.normal(0, 0.5, 100)

#     # 调用绘图函数
#     plot_Scatter_plot_graph(x, y)
















# # /////////////////////////计算指标与绘制散点图////////////////////
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from skimage.metrics import structural_similarity as ssim
# import numpy as np
# import matplotlib.pyplot as plt


# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from skimage.metrics import structural_similarity as ssim
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# def calculate_metrics(true_values, predicted_values):
#     mse = mean_squared_error(true_values, predicted_values)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(true_values, predicted_values)
#     mre = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
#     mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    
#     # Calculating SSIM with data_range specified
#     ssim_value, _ = ssim(true_values, predicted_values, data_range=predicted_values.max() - predicted_values.min(), full=True)
    
#     return mse, rmse, mae, mre, mape, ssim_value

# def plot_Scatter_plot_graph(true_values, predicted_values):
#     true_values_flat = true_values.flatten()
#     predicted_values_flat = predicted_values.flatten()
    
#     # Linear regression model for fit line
#     model = LinearRegression()
#     true_values_reshaped = true_values_flat.reshape(-1, 1)
#     model.fit(true_values_reshaped, predicted_values_flat)
#     fit_line = model.predict(true_values_reshaped)
#     slope = model.coef_[0]
    
#     # Calculate metrics including SSIM
#     mse, rmse, mae, mre, mape, ssim_value = calculate_metrics(true_values, predicted_values)
#     n_points = len(true_values_flat)
    
#     # Plotting
#     plt.figure(figsize=(8, 8))
#     plt.scatter(true_values_flat, predicted_values_flat, alpha=0.5, label='Data Points')
#     plt.plot([true_values_flat.min(), true_values_flat.max()], [true_values_flat.min(), true_values_flat.max()], 'r--', label='1:1 Line')
#     plt.plot(true_values_flat, fit_line, 'b-', label=f'Fit Line (Slope = {slope:.2f})')
    
#     # Adding metrics including SSIM
#     metrics_text = (f'N = {n_points}\n'
#                     f'MSE = {mse:.3f}\n'
#                     f'RMSE = {rmse:.3f}\n'
#                     f'MAE = {mae:.3f}\n'
#                     f'MRE = {mre:.2f}%\n'
#                     f'MAPE = {mape:.2f}%\n'
#                     f'SSIM = {ssim_value:.3f}\n'
#                     f'Slope = {slope:.3f}')
#     plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
#              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.5))
    
#     plt.xlabel('True Values °C')
#     plt.ylabel('Predicted Values °C')
#     plt.title('Scatter plot')
#     plt.legend(loc='lower right')
#     plt.grid(True)
#     plt.show()






# 获取指定区域的指定天数的栅格数组

# 获取指定时间索引的栅格数组
def get_sst_data_for_day_and_area(file_path, day, lat_range, lon_range):
    # 打开NC文件
    dataset = nc.Dataset(file_path)

    # 提取经纬度数据
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]
    
    # 定位指定经纬度范围的索引
    lon_min_idx = np.searchsorted(lons, lon_range[0])
    lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
    lat_min_idx = np.searchsorted(lats, lat_range[0])
    lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

    # 获取指定范围和天数的海表温度数据
    sst_data = dataset.variables['analysed_sst'][day, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
    
    # 处理无效值
    if hasattr(dataset.variables['analysed_sst'], '_FillValue'):
        fill_value = dataset.variables['analysed_sst']._FillValue
        sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data).filled(np.nan)  # 使用np.nan填充无效值
    else:
        sst_data_masked = np.ma.masked_invalid(sst_data).filled(np.nan)

    # 关闭文件
    dataset.close()
    
    # 将温度从开尔文转换为摄氏度
    sst_data_masked = sst_data_masked - 273.15

    return sst_data_masked









# 加载NC文件的全部空间范围的栅格
def load_and_prepare_data(nc_file_path):
    """Load and prepare SST data from a NetCDF file."""
    ds = nc.Dataset(nc_file_path)
    sst = ds.variables['sst'][:]  # Assuming the dimensions are (time, lat, lon)
    sst = np.nan_to_num(sst, nan=np.nan)
    # 检查并处理异常值
    sst = np.clip(sst, -1e10, 1e10)  # 将值限制在合理范围内
    return sst

# # 读入NC文件，指定日期绘制指定范围的栅格函数
# def plot_sst_for_days(file_path, days, lat_range, lon_range):
#     # 打开NC文件
#     dataset = nc.Dataset(file_path)

#     # 提取经纬度数据
#     lons = dataset.variables['lon'][:]
#     lats = dataset.variables['lat'][:]
    
#     # 定位指定经纬度范围的索引
#     lon_min_idx = np.searchsorted(lons, lon_range[0])
#     lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
#     lat_min_idx = np.searchsorted(lats, lat_range[0])
#     lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

#     # 遍历指定的天数
#     for day in days:
#         # 获取指定范围和天数的海表温度数据
#         sst_data = dataset.variables['sst'][day, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
        
#         # 处理无效值
#         if hasattr(dataset.variables['sst'], '_FillValue'):
#             fill_value = dataset.variables['sst']._FillValue
#             sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data)
#         else:
#             sst_data_masked = np.ma.masked_invalid(sst_data)

#         # 准备绘图
#         plt.figure(figsize=(12, 6))

#         # 使用imshow绘制数据
#         plt.imshow(sst_data_masked, origin='lower', cmap='jet', 
#                    extent=[lons[lon_min_idx], lons[lon_max_idx-1], lats[lat_min_idx], lats[lat_max_idx-1]])

#         # 添加色标
#         plt.colorbar(label='Sea Surface Temperature (°C)')

#         # 添加标题和轴标签
#         plt.title(f'Sea Surface Temperature (SST) for day {day+1}')
#         plt.xlabel('Longitude')
#         plt.ylabel('Latitude')
        
#         plt.show()
# file = 'sst.day.mean.2006.nc'
# plot_sst_for_days(file, [4,5], (0,30), (160,180))

import netCDF4 as nc
import numpy as np
# 将NC文件当中的指定时间范围的指定经纬度范围的图像返回栅格， 获取NC文件的第几张到第几张，而不是对应的时间索引
def get_sst_data_for_time_range_and_area(file_path, start_day, end_day, lat_range, lon_range):
    # 打开NC文件
    dataset = nc.Dataset(file_path)

    # 提取经纬度数据
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]

    # 定位指定经纬度范围的索引
    lon_min_idx = np.searchsorted(lons, lon_range[0])
    lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
    lat_min_idx = np.searchsorted(lats, lat_range[0])
    lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

    # 提取指定时间范围的海表温度数据
    sst_data_list = []
    for day in range(start_day, end_day + 1):
        sst_data = dataset.variables['analysed_sst'][day, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
        
        # 处理无效值
        if hasattr(dataset.variables['analysed_sst'], '_FillValue'):
            fill_value = dataset.variables['analysed_sst']._FillValue
            sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data).filled(np.nan)  # 使用np.nan填充无效值
        else:
            sst_data_masked = np.ma.masked_invalid(sst_data).filled(np.nan)

        # 将温度从开尔文转换为摄氏度
        sst_data_masked = sst_data_masked - 273.15

        sst_data_list.append(sst_data_masked)

    # 关闭文件
    dataset.close()

    return np.array(sst_data_list)



# 打开NC文件，可视化处理指定时间范围内指定空间范围的海温栅格
def plot_sst_for_days(file_path, days, lat_range, lon_range):
    # 打开NC文件
    dataset = nc.Dataset(file_path)

    # 提取经纬度数据
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]
    
    # 定位指定经纬度范围的索引
    lon_min_idx = np.searchsorted(lons, lon_range[0])
    lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
    lat_min_idx = np.searchsorted(lats, lat_range[0])
    lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

    # 遍历指定的天数
    for day in days:
        # 获取指定范围和天数的海表温度数据
        sst_data = dataset.variables['analysed_sst'][day, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
        
        # 处理无效值
        if hasattr(dataset.variables['analysed_sst'], '_FillValue'):
            fill_value = dataset.variables['analysed_sst']._FillValue
            sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data)-273.15
        else:
            sst_data_masked = np.ma.masked_invalid(sst_data)-273.15

        # 准备绘图
        plt.figure(figsize=(12, 6))

        # 使用imshow绘制数据
        plt.imshow(sst_data_masked, origin='lower', cmap='jet', 
                   extent=[lons[lon_min_idx], lons[lon_max_idx-1], lats[lat_min_idx], lats[lat_max_idx-1]])

        # 添加色标
        plt.colorbar(label='Sea Surface Temperature (°C)')

        # 添加标题和轴标签
        plt.title(f'Sea Surface Temperature (SST) for day {day+1}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()



# 打印位置索引函数
def print_grid_indices_at_time(nc_file_path, day_index, lat_range, lon_range):
    # 打开NC文件
    dataset = nc.Dataset(nc_file_path)

    # 获取经纬度数据
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]

    # 定位指定经纬度范围的索引
    lon_min_idx = np.searchsorted(lons, lon_range[0])
    lon_max_idx = np.searchsorted(lons, lon_range[1], side='right')
    lat_min_idx = np.searchsorted(lats, lat_range[0])
    lat_max_idx = np.searchsorted(lats, lat_range[1], side='right')

    # 打印结果
    print(f"在时间索引 {day_index}，经纬度范围 {lat_range} 到 {lon_range} 内的格点索引为：")
    print(f"经度索引范围：{lon_min_idx} 到 {lon_max_idx - 1}")
    print(f"纬度索引范围：{lat_min_idx} 到 {lat_max_idx - 1}")

    # 可以选择性地返回这些索引，如果需要进一步处理或测试
    return (lon_min_idx, lon_max_idx - 1), (lat_min_idx, lat_max_idx - 1)

# # 示例调用
# nc_file_path = 'sst.day.mean.2006.nc'
# day_index = 10
# lat_range = (28, 29)  # 示例纬度范围
# lon_range = (170, 171)  # 示例经度范围
# print_grid_indices_at_time(nc_file_path, day_index, lat_range, lon_range)



# 普通的版本计算散点图和计算指标
def calculate_error_metrics(true_values, predicted_values):
    """
    计算并打印MSE、RMSE和MAE。
    """
    # 确保输入的真实值和预测值形状一致，并且去除NaN值
    mask = ~np.isnan(true_values) & ~np.isnan(predicted_values)
    true_values = true_values[mask]
    predicted_values = predicted_values[mask]
    
    # 计算MSE
    mse = np.mean((true_values - predicted_values) ** 2)
    # 计算RMSE
    rmse = np.sqrt(mse)
    # 计算MAE
    mae = np.mean(np.abs(true_values - predicted_values))
    
    print(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")


def plot_consistency_graph(true_values, predicted_values):
    """
    绘制预测值与实际值的一致性图。
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--')  # 绘制1:1线
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Consistency Graph')
    plt.grid(True)
    plt.show()


# # 输入NC文件路径    打印NC文件变量信息
def print_myNC_information(file):
  
# 打开NC文件
    dataset = nc.Dataset(file)

# 获取所有变量的键
    all_vars = dataset.variables.keys()
    print("Variables in the dataset:", all_vars)

# 获取所有变量的信息
    all_vars_info = list(dataset.variables.items())
    print("\nVariables information:")
    for var in all_vars_info:
       print(var[0], var[1])
# 测试用例
# file = '20060129-REMSS-L4HRfnd-GLOB-v01-fv01-mw_ir_OI.nc'
# print_myNC_information(file)


# 输入NC文件显示指定某一天时间的指定范围栅格
def plot_sst_for_date_and_region(nc_file, date_index, lon_range, lat_range):
    """
    绘制NetCDF文件中特定日期和区域的海表温度(SST)。

    参数:
    - nc_file: str, NetCDF文件的路径。
    - date_index: int, 数据集中日期的索引。
    - lon_range: tuple, 经度范围（最小值, 最大值）。
    - lat_range: tuple, 纬度范围（最小值, 最大值）。
    """
    # 打开NetCDF文件
    dataset = nc.Dataset(nc_file)

    # 提取经纬度数据
    lons = dataset.variables['lon'][:]
    lats = dataset.variables['lat'][:]

    # 定位指定经纬度范围的索引
    lon_min_idx = np.where(lons >= lon_range[0])[0][0]
    lon_max_idx = np.where(lons <= lon_range[1])[0][-1]
    lat_min_idx = np.where(lats >= lat_range[0])[0][0]
    lat_max_idx = np.where(lats <= lat_range[1])[0][-1]

    # 获取指定日期和范围内的海表温度数据
    sst_data = dataset.variables['analysed_sst'][date_index, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]

    # 检查是否有_fillValue属性，若有则用它来掩盖无效数据
    if hasattr(dataset.variables['analysed_sst'], '_FillValue'):
        fill_value = dataset.variables['analysed_sst']._FillValue
        sst_data_masked = np.ma.masked_where(sst_data == fill_value, sst_data)
    else:
        sst_data_masked = np.ma.masked_invalid(sst_data)

    sst_data_masked=sst_data_masked-273.15#这一句使用新数据集需要，温度转换
    # 准备绘图
    plt.figure(figsize=(10, 6))
    plt.imshow(sst_data_masked, origin='lower', cmap='jet', 
               extent=[lons[lon_min_idx], lons[lon_max_idx], lats[lat_min_idx], lats[lat_max_idx]])
    plt.colorbar(label='海表温度 (°C)')
    plt.title(f'第{date_index}天指定区域的海表温度 (SST)')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.show()

# 示例用法:
# file_path = '20060129-REMSS-L4HRfnd-GLOB-v01-fv01-mw_ir_OI.nc'
# plot_sst_for_date_and_region(file_path, 0, (170, 175), (28, 32))




def print_grid_shape(interpolated_grid):
    """
    打印插值栅格图像的尺寸（行数和列数）。

    参数:
    interpolated_grid -- 插值后的栅格图像，一个NumPy数组。
    """
    rows, cols = interpolated_grid.shape
    print(f"栅格图像的尺寸是 {rows} 行 {cols} 列。")

import numpy as np
import matplotlib.pyplot as plt

# def visualize_displacement_field(displacement_field, step=10, scale=1, quiverkey=False):
#     """
#     可视化位移场。
    
#     参数:
#     - displacement_field (numpy.ndarray): 位移场数组，形状应为 (height, width, 2)。
#     - step (int): 矢量显示的步长，数字越大，显示的矢量越稀疏。
#     - scale (float): 控制矢量长度的比例因子，数字越大矢量越短。
#     - quiverkey (bool): 是否在图中添加矢量长度说明。
#     """
#     height, width, _ = displacement_field.shape
#     Y, X = np.mgrid[0:height:step, 0:width:step]
#     U = displacement_field[::step, ::step, 1]
#     V = displacement_field[::step, ::step, 0]

#     fig, ax = plt.subplots(figsize=(10, 10))
#     quiver = ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scale, color='red')
    
#     if quiverkey:
#         # 添加矢量长度说明
#         plt.quiverkey(quiver, X=0.9, Y=1.05, U=10,
#                       label='Quiver key, length = 10', labelpos='E')

#     plt.title("Displacement Field Visualization")
#     plt.xlabel('Pixel X coordinate')
#     plt.ylabel('Pixel Y coordinate')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.show()

# # 使用示例
# # displacement_field = calculate_displacement(image1, image2, window_size=3, step=1)
# # visualize_displacement_field(displacement_field, step=20, scale=20, quiverkey=True)


# 输入栅格和经纬度进行可视化
# def plot_raster_grid(raster_data, lat_range, lon_range):
#     """
#     绘制栅格数据的函数。

#     参数:
#     raster_data : np.array
#         栅格数据数组。
#     lat_range : tuple
#         经度范围，形式为(min_lat, max_lat)。
#     lon_range : tuple
#         纬度范围，形式为(min_lon, max_lon)。
#     """
#     # 创建图像
#     plt.figure(figsize=(12, 6))

#     # 绘制栅格图
#     plt.imshow(raster_data, origin='lower', cmap='jet', extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]])

#     # 添加色标
#     plt.colorbar(label='Value')

#     # 设置标题和轴标签
#     plt.title('Raster Grid Visualization')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
    
#     # 显示图像
#     plt.show()



# raster_data = np.random.rand(4, 4)
# # 指定的经纬度范围
# lat_range = (28, 29)
# lon_range = (170, 171)
# plot_raster_grid(raster_data, lat_range, lon_range)



from matplotlib.colors import Normalize
def plot_raster_grid(raster_data, lat_range, lon_range, vmin=None, vmax=None, title=None):
    """
    绘制栅格数据的函数。

    参数:
    raster_data : np.array
        栅格数据数组。
    lat_range : tuple
        纬度范围，形式为(min_lat, max_lat)。
    lon_range : tuple
        经度范围，形式为(min_lon, max_lon)。
    vmin : float, optional
        色带最小值。
    vmax : float, optional
        色带最大值。
    title : str, optional
        图像标题。
    """
    # 创建图像
    plt.figure(figsize=(12, 6))

    # 绘制栅格图
    plt.imshow(
        raster_data,
        origin='lower',
        cmap='jet',
        extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
        norm=Normalize(vmin=vmin, vmax=vmax)
    )

    # 添加色标
    plt.colorbar(label='Value')

    # 设置标题和轴标签
    if title is not None:
        plt.title(title)
    else:
        plt.title('Raster Grid Visualization')
    plt.xlabel('Longitude(°E)')
    plt.ylabel('Latitude(°N)')
    
    # 显示图像
    plt.show()


    

def print_grid_point_value(images, time_index, lat_idx, lon_idx):
    """
    打印指定时间步和格点的值。

    参数:
    images : np.array
        包含SST数据的数组，维度为(time, lat, lon)。
    time_index : int
        时间步的索引。
    lat_idx : int
        纬度的索引。
    lon_idx : int
        经度的索引。
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

# 示例使用
# nc_file_path = 'sst.day.mean.2006.nc'
# images = load_and_prepare_data(nc_file_path)
# print_grid_shape(images[0])

# # # 打印指定时间步和格点的值
# time_index = 4
# lat_idx = 112
# lon_idx = 680
# print_grid_point_value(images, time_index, lat_idx, lon_idx)



import netCDF4 as nc
import numpy as np

import netCDF4 as nc
# 输出NC文件头文件
def check_nc_file_structure(nc_file_path):
    """检查 NetCDF 文件的结构"""
    ds = nc.Dataset(nc_file_path)
    print(ds)
    print(ds.variables.keys())
    for var in ds.variables:
        print(f"{var}: {ds.variables[var].dimensions}, {ds.variables[var].shape}")

# 示例使用
# nc_file_path = 'sst.day.mean.2006.nc'
# check_nc_file_structure(nc_file_path)



import netCDF4 as nc
import numpy as np

def get_lat_lon_indices(ds, lat_range, lon_range):
    """将经纬度范围转换为索引范围"""
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    min_lat_idx = np.abs(lat - lat_range[0]).argmin()
    max_lat_idx = np.abs(lat - lat_range[1]).argmin()
    min_lon_idx = np.abs(lon - lon_range[0]).argmin()
    max_lon_idx = np.abs(lon - lon_range[1]).argmin()
    return (min_lat_idx, max_lat_idx), (min_lon_idx, max_lon_idx)

def print_nc_grid_range(nc_file_path, time_index, lat_range, lon_range):
    """
    打印 NetCDF 文件中指定时间步和经纬度范围内的 SST 值。

    参数:
    nc_file_path : str
        NetCDF 文件路径。
    time_index : int
        时间步的索引。
    lat_range : tuple
        纬度范围，形式为 (min_lat, max_lat)。
    lon_range : tuple
        经度范围，形式为 (min_lon, max_lon)。
    """
    ds = nc.Dataset(nc_file_path)
    sst = ds.variables['sst'][:]  # 假设 SST 数据的变量名是 'sst'

    # 将掩码数组转换为 NaN
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

# # 示例使用
# nc_file_path = 'sst.day.mean.2006.nc'

# # 打印指定时间步和经纬度范围的栅格值
# time_index = 4
# lat_range = (28, 29)  # 示例纬度范围
# lon_range = (170, 171)  # 示例经度范围
# print_nc_grid_range(nc_file_path, time_index, lat_range, lon_range)



# 可视化位移场
def plot_displacement_field(displacement_field):
    """Plot displacement field as vector field"""
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
    打印NetCDF文件中的时间索引及其对应的实际日期时间。
    
    参数:
    - nc_file_path: NC文件的路径。
    """
    with nc.Dataset(nc_file_path, 'r') as ds:
        # 获取时间变量
        time_var = ds.variables['time']
        time_units = time_var.units
        time_calendar = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'
        
        # 将时间索引转换为实际日期时间
        time_indices = time_var[:]
        dates = num2date(time_indices, units=time_units, calendar=time_calendar)
        
        # 打印时间索引及其对应的实际日期时间
        for idx, date in zip(time_indices, dates):
            print(f"Index: {idx}, Date: {date}")

# 示例使用
  # 这里替换为你的NC文件路径
# print_nc_file_time_indices(nc_file_path)


# 打印显示NC文件在指定时刻和指定格点的SST值
def print_sst_at_point_and_time(nc_file_path, time_index, lat_target, lon_target):
    """
    打印NC文件在指定时刻和指定格点的海表面温度值。

    参数:
    - nc_file_path: NC文件的路径。
    - time_index: 指定时刻的索引。
    - lat_target: 指定格点的纬度。
    - lon_target: 指定格点的经度。
    """
    # 打开NC文件
    dataset = nc.Dataset(nc_file_path)

    # 获取变量
    sst = dataset.variables['sst']  # 假设变量名为 'sst'
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]

    # 找到最接近目标纬度和经度的索引
    lat_idx = (np.abs(lats - lat_target)).argmin()
    lon_idx = (np.abs(lons - lon_target)).argmin()

    # 获取并打印指定时刻和格点的SST值
    sst_value = sst[time_index, lat_idx, lon_idx]
    print(f"SST at time index {time_index}, lat {lat_target}, lon {lon_target}: {sst_value}°C")

