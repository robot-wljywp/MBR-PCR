import open3d as o3d
import numpy as np
import json, os
from tqdm import tqdm
from tools.metric import translation_error, rotation_error
import torch, faiss
import turboreg_gpu  
from tools.ground import cut_pc_by_terrain
from utils.distributionSample.build import distributionSample
import copy

def calTeAndRe(T1, T2):
    return translation_error(T1[:3, 3], T2[:3, 3]), rotation_error(T1[:3, :3], T2[:3, :3])

def calTeAndRe(T1, T2):
    return translation_error(T1[:3, 3], T2[:3, 3]), rotation_error(T1[:3, :3], T2[:3, :3])

def grid_filter(points, threshold = 2, resolution=0.3):
    x = points[:, 0]
    y = points[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    width = int((x_max - x_min) / resolution) + 1
    height = int((y_max - y_min) / resolution) + 1

    # 将点云投影到xy平面栅格中
    col_indices = np.floor((x - x_min) / resolution).astype(int)
    row_indices = np.floor((y - y_min) / resolution).astype(int)

    # 创建栅格并统计每个栅格中的点数
    grid_indices = row_indices * width + col_indices  # 将二维索引转换为一维索引
    valid_indices = (col_indices >= 0) & (col_indices < width) & \
                    (row_indices >= 0) & (row_indices < height)

    # 统计每个栅格中的点数
    grid_counts = np.bincount(grid_indices[valid_indices], minlength=width * height)

    # 合并两个布尔索引
    combined_mask = valid_indices & (grid_counts[grid_indices] >= threshold)

    return combined_mask

def bidirectional_feature_match_and_align(points1, features1, points2, features2):

    N1, D1 = features1.shape
    N2, D2 = features2.shape

    # 确保特征为 float32 (Faiss 要求)
    features1 = features1.astype('float32')
    features2 = features2.astype('float32')

    index = faiss.IndexFlatL2(D1)  # 使用 L2 距离
    index.add(features2)           # 将目标特征加入索引

    distances_1to2, indices_1to2 = index.search(features1, k=1)  # (N1, 1), (N1, 1)
    indices_1to2 = indices_1to2.flatten()  # (N1,)
    distances_1to2 = distances_1to2.flatten()  # (N1,)

    index_rev = faiss.IndexFlatL2(D1)
    index_rev.add(features1)

    distances_2to1, indices_2to1 = index_rev.search(features2, k=1)  # (N2, 1), (N2, 1)
    indices_2to1 = indices_2to1.flatten()  # (N2,)
    distances_2to1 = distances_2to1.flatten()  # (N2,)

    mutual_matches = []
    for i in range(N1):
        j = indices_1to2[i]  # 点1[i] 在点2中的最近邻
        if indices_2to1[j] == i:  # 检查点2[j] 的最近邻是否是点1[i]
            mutual_matches.append((i, j, distances_1to2[i]))  # 存储 (i, j, distance)

    idx1_aligned = [match[0] for match in mutual_matches]  # 点云1的索引
    idx2_aligned = [match[1] for match in mutual_matches]  # 点云2的索引

    # 根据索引提取对齐后的点云坐标
    aligned_points1 = points1[idx1_aligned]  # (M, 3)
    aligned_points2 = points2[idx2_aligned]  # (M, 3)

    return aligned_points1, aligned_points2

def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) == 3:
            color = np.repeat(np.array(color)[np.newaxis, ...], xyz.shape[0], axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def find_most_similar_segment(vec1, vec2):
    n = len(vec1) 
    max_corr = -1.0
    best_start = 0
    best_length = n
    start = 0
    for length in range(3, n + 1):
        end = start + length
        
        segment1 = vec1[start:end]
        segment2 = vec2[start:end]
        
        corr = np.corrcoef(segment1, segment2)[0, 1]
        
        if corr > max_corr:
            max_corr = corr
            best_start = start
            best_length = length
    
    best_end = best_start + best_length

    return best_start, best_end, max_corr

def normalize_vectors_jointly(vec1, vec2):
    combined = np.concatenate((vec1, vec2))
    global_min = combined.min()
    global_max = combined.max()

    if global_max == global_min:
        return np.full_like(vec1, 0.5), np.full_like(vec2, 0.5)
    
    norm_vec1 = (vec1 - global_min) / (global_max - global_min)
    norm_vec2 = (vec2 - global_min) / (global_max - global_min)
    
    return norm_vec1, norm_vec2

def merge_first_n_arrays(array_list, n):
    if not array_list:
        return []
    
    if n <= 0:
        return [arr.copy() for arr in array_list]
    
    if n >= len(array_list):
        merged = []
        for arr in array_list:
            merged.extend(arr.tolist())
        return merged
    
    merged_part = []
    for i in range(n+1):
        merged_part.extend(array_list[i].tolist())

    return merged_part

def MBR_PCR(points1:np.ndarray, points2:np.ndarray, features1:np.ndarray, features2:np.ndarray)-> np.ndarray:
    points1_copy = copy.deepcopy(points1)
    points2_copy = copy.deepcopy(points2)

    ####################################
    # 根据地形切割点云
    ####################################
    pcd1, pcd1_nums, layer_list1 = cut_pc_by_terrain(pointcloud_data=points1)
    pcd2, pcd2_nums, layer_list2 = cut_pc_by_terrain(pointcloud_data=points2)
    
    ####################################
    # 比较两个分布之间的最大部分，并提取点云
    ####################################
    pcd1_nums, pcd2_nums = normalize_vectors_jointly(pcd1_nums, pcd2_nums)
    _, end_, _ = find_most_similar_segment(pcd1_nums, pcd2_nums)
    layer_list_sum1 = merge_first_n_arrays(layer_list1, end_ + 1)
    layer_list_sum2 = merge_first_n_arrays(layer_list2, end_ + 1)

    index1 = layer_list_sum1
    index2 = layer_list_sum2

    points1, features1 = points1[index1], features1[index1]
    points2, features2 = points2[index2], features2[index2]

    aligned_p1, aligned_p2 = bidirectional_feature_match_and_align(
        points1, features1, points2, features2
    )
    aligned_p1 = torch.from_numpy(aligned_p1).cuda().float()
    aligned_p2 = torch.from_numpy(aligned_p2).cuda().float()

    reger = turboreg_gpu.TurboRegGPU(
        3000,      # max_N: Maximum number of correspondences
        1,     # tau_length_consis: \tau (consistency threshold for feature length/distance)
        1000,    # num_pivot: Number of pivot points, K_1
        0.8,      # radiu_nms: Radius for avoiding the instability of the solution
        4,       # tau_inlier: Threshold for inlier points. NOTE: just for post-refinement (REF@PointDSC/SC2PCR/MAC)
        "IN"       # eval_metric: MetricType (e.g., "IN" for Inlier Number, or "MAE" / "MSE")
    )
    trans = reger.run_reg(aligned_p1, aligned_p2, torch.from_numpy(points1_copy).cuda().float(), torch.from_numpy(points2_copy).cuda().float())
    trans = trans.detach().cpu().numpy().astype(np.float64)

    new_pc1 = make_open3d_point_cloud(points1_copy).transform(trans)
    new_pc2 = make_open3d_point_cloud(points2_copy)

    new_pc1_np = np.array(new_pc1.points)
    new_pc2_np = np.array(new_pc2.points)

    pc1And2 = np.concatenate([new_pc1_np, new_pc2_np], axis = 0)
    pc1And2Min = np.min(pc1And2, axis = 0)
    pc1And2Max = np.max(pc1And2, axis = 0)
    (pc1_, pc2_) = distributionSample.distributionSample(new_pc1_np, new_pc2_np, pc1And2Min, pc1And2Max, 10)

    new_pc1 = make_open3d_point_cloud(pc1_)
    new_pc2 = make_open3d_point_cloud(pc2_)
    trans_ = o3d.pipelines.registration.registration_icp(
        new_pc1,
        new_pc2,
        1.5,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
    trans = trans_.transformation @ trans

    return trans

if __name__ == '__main__':
    ############################################################
    # PARAMETERS
    ############################################################
    datasets = 'CampHill' # CampHill or GrAco
    mode = 'cross' # same or cross
    print(f'Dataset is {datasets}, mode is {mode}')

    ############################################################
    # LOADING POINT CLOUDS
    ############################################################
    json_path = f'Datasets/{datasets}_json/{mode}_source_pc_registration.json'
    with open(json_path, 'r') as f:
        json_list = json.load(f)
    states = []
    feature_path = f'Datasets/{datasets}'

    for j in tqdm(json_list):

        pcd1_index = json_list[j]['pcd1_index']
        pcd2_index = json_list[j]['pcd2_index']

        pc1_path = os.path.join(feature_path, f'{pcd1_index:06d}', 'pointcloud.npy')
        pc2_path = os.path.join(feature_path, f'{pcd2_index:06d}', 'pointcloud.npy')
        f1_path = os.path.join(feature_path, f'{pcd1_index:06d}', 'point_features.npy')
        f2_path = os.path.join(feature_path, f'{pcd2_index:06d}', 'point_features.npy')
        
        points1 = np.load(pc1_path)
        features1 = np.load(f1_path)
        points2 = np.load(pc2_path)
        features2 = np.load(f2_path)

        trans = MBR_PCR(points1=points1, points2=points2, features1=features1, features2=features2)

        TE, RE = calTeAndRe(trans, np.array(json_list[j]['trans']))
        states.append(np.array([TE < 1 and RE < 10, TE, RE]))

    states = np.array(states)
    Recall = states[:, 0].sum() / states.shape[0]
    TE = states[states[:, 0] == 1, 1].mean()
    RE = states[states[:, 0] == 1, 2].mean()
    print('Recall of DGR: {:.2f}'.format(Recall*100))
    print('TE of DGR: {:.2f}'.format(TE*100))
    print('RE of DGR: {:.2f}'.format(RE))
