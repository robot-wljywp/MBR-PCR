import open3d as o3d
import numpy as np
from scipy import ndimage
from skimage.transform import resize

def pcd_to_25d_grid(pcd, resolution=0.5):

    points = np.asarray(pcd.points)
    
    x_min, y_min = -25, -25
    x_max, y_max = 25, 25
    
    nx = int(np.ceil((x_max - x_min) / resolution)) + 1
    ny = int(np.ceil((y_max - y_min) / resolution)) + 1
    
    xi = ((points[:, 0] - x_min) / resolution).astype(int)
    yi = ((points[:, 1] - y_min) / resolution).astype(int)
    
    valid_mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    xi_valid = xi[valid_mask]
    yi_valid = yi[valid_mask]
    z_valid = points[valid_mask, 2]

    indices_1d = xi_valid * ny + yi_valid
    
    min_z = np.full(nx*ny, np.inf)
    np.minimum.at(min_z, indices_1d, z_valid)

    count = np.bincount(indices_1d, minlength=nx*ny)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        height_map_flat = np.where(count > 0, min_z / count, np.nan)
    
    height_map = height_map_flat.reshape((nx, ny))

    if np.isnan(height_map).any():
        mask = ~np.isnan(height_map)
        indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        height_map = height_map[tuple(indices)]
    
    return x_min, x_max, y_min, y_max, resolution, height_map

def fill_in_circle_only(data, fill_value=np.nan):
    H, W = data.shape
    center = (H / 2 - 0.5, W / 2 - 0.5)
    radius = min(H, W) / 2
    
    # 快速创建圆形掩码
    y, x = np.ogrid[:H, :W]
    circle_mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    
    if not np.any(circle_mask):
        return np.full_like(data, fill_value), circle_mask
    
    result = np.full_like(data, fill_value)
    
    valid_in_circle = ~np.isnan(data) & circle_mask
    if np.any(valid_in_circle):
        result[valid_in_circle] = data[valid_in_circle]
    
    nan_in_circle = np.isnan(data) & circle_mask
    if not np.any(nan_in_circle):
        return result, circle_mask
    
    indices = ndimage.distance_transform_edt(
        nan_in_circle,
        return_distances=False,
        return_indices=True
    )
    
    filled_values = data[tuple(indices)]
    result[nan_in_circle] = filled_values[nan_in_circle]
    
    return result, circle_mask

def progressive_morphological_filtering(height_map, resolution, max_window=5.0, slope=0.1, initial_distance=0.1, max_distance=3.0):

    window_sizes = []
    current_size = 0.7
    while current_size <= max_window:
        window_sizes.append(current_size)
        current_size *= 2
    
    kernel_sizes = [max(3, int(np.ceil(w / resolution)) | 1) for w in window_sizes]  # 确保奇数
    
    ground_mask = np.zeros_like(height_map, dtype=bool)
    temp_map = height_map.astype(np.float32)  
    
    for i, window in enumerate(window_sizes):
        dist_threshold = min(initial_distance + slope * window, max_distance)
        kernel_size = kernel_sizes[i]
        
        if kernel_size > 10:
            temp_min = ndimage.minimum_filter1d(temp_map, kernel_size, axis=1, mode='nearest')
            opened = ndimage.minimum_filter1d(temp_min, kernel_size, axis=0, mode='nearest')
            opened = ndimage.maximum_filter1d(opened, kernel_size, axis=1, mode='nearest')
            opened = ndimage.maximum_filter1d(opened, kernel_size, axis=0, mode='nearest')
        else:
            opened = ndimage.grey_opening(temp_map, size=(kernel_size, kernel_size))
        
        diff = height_map - opened
        new_ground = (diff <= dist_threshold) & ~ground_mask
        
        ground_mask |= new_ground
        temp_map[new_ground] = np.nan
        
    return ground_mask

def extract_ground(pcd, resolution=0.5, max_window=5.0, grid_size = 3, slope=0.3, initial_distance=0.15, max_distance=2.0):
    x_min, x_max, y_min, y_max, res, height_map = pcd_to_25d_grid(pcd, resolution=resolution)

    ground_mask = progressive_morphological_filtering(
        height_map, 
        resolution=resolution, 
        max_window=max_window, 
        slope=slope, 
        initial_distance=initial_distance, 
        max_distance=max_distance
    )

    # 获取原始点云坐标
    points = np.asarray(pcd.points)
    ny, nx = height_map.shape[1], height_map.shape[0]
    xi = ((points[:, 0] - x_min) / res).astype(int)
    yi = ((points[:, 1] - y_min) / res).astype(int)
    
    valid_mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    
    # 对于有效索引，检查ground_mask
    valid_xi = xi[valid_mask]
    valid_yi = yi[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    is_ground = ground_mask[valid_xi, valid_yi]
    ground_indices = valid_indices[is_ground]

    ground_pcd = pcd.select_by_index(ground_indices)
    non_ground_pcd = 0

    x_min, x_max, y_min, y_max, _, height_map = pcd_to_25d_grid(ground_pcd, resolution=resolution)
    kernel_size = int(grid_size / resolution) + 3
    height_map = ndimage.grey_erosion(height_map, size=(kernel_size, kernel_size))
    height_map = resize(height_map, (50, 50), anti_aliasing=True, preserve_range=False)
    height_map, _ = fill_in_circle_only(height_map, fill_value=np.nan)

    return ground_pcd, non_ground_pcd, is_ground, height_map

def cut_pc_by_terrain(pointcloud_data:np.array):
    resolution_ = 0.15
    grid_size = 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_data)
    if not pcd.has_points():
        raise ValueError("点云为空！")

    ground, non_ground, ground_mask, height_map = extract_ground(
        pcd,
        resolution=resolution_,          
        max_window=0.8,           
        grid_size = grid_size,
        slope=0.5,                
        initial_distance=0.02,    
        max_distance=1         
    )

    layer_bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

    # 提取坐标
    x_coords = pointcloud_data[:, 0]
    y_coords = pointcloud_data[:, 1]
    z_coords = pointcloud_data[:, 2]

    cell_size = 1 

    col_indices = ((x_coords - (-25)) // cell_size).astype(int)
    row_indices = ((y_coords - (-25)) // cell_size).astype(int)

    # 判断索引是否在有效范围内
    valid_mask = (
        (col_indices >= 0) & (col_indices < height_map.shape[0]) &
        (row_indices >= 0) & (row_indices < height_map.shape[1])
    )

    assert np.all(valid_mask)
    pointcloud_data = pointcloud_data[valid_mask]
    z_coords = z_coords[valid_mask]
    row_indices = row_indices[valid_mask]
    col_indices = col_indices[valid_mask]

    # 获取对应地面高程
    ground_z = height_map[col_indices, row_indices]

    # 计算离地高度
    height_above_ground = z_coords - ground_z

    within_range_mask = height_above_ground <= layer_bounds[-1]
    final_points = pointcloud_data[within_range_mask]
    final_hag = height_above_ground[within_range_mask]

    layers = []
    point_nums = []
    indexs = []
    for i in range(len(layer_bounds) - 1):
        lower = layer_bounds[i]
        upper = layer_bounds[i + 1]
        layer_mask = (final_hag >= lower) & (final_hag < upper)
        layer_points = final_points[layer_mask]
        indexs.append(np.where(within_range_mask)[0][np.where(layer_mask)[0]])

        layer_pcd = o3d.geometry.PointCloud()
        layer_pcd.points = o3d.utility.Vector3dVector(layer_points)
        point_nums.append(layer_points.shape[0])
        layers.append(layer_pcd)

    return layers, point_nums, indexs
