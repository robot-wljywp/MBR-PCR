import numpy as np
import sys
import json

def hit(rank, ground_truth):
    # HR is equal to Recall when dataset is loo split.
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0
    return result


def precision(rank, ground_truth):
    # Precision is meaningless when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / np.arange(1, len(rank) + 1)
    return result


def recall(rank, ground_truth):
    # Recall is equal to HR when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / len(ground_truth)
    return result


def map(rank, ground_truth):
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    # relevant_num = np.cumsum([1 if item in ground_truth else 0 for item in rank])
    relevant_num = np.cumsum([min(idx + 1, len(ground_truth)) for idx, _ in enumerate(rank)])
    result = [p / r_num if r_num != 0 else 0 for p, r_num in zip(sum_pre, relevant_num)]
    return result


def ndcg(rank, ground_truth):
    len_rank = len(rank)
    idcg_len = min(len(ground_truth), len_rank)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]

    dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg / idcg
    return result


def mrr(rank, ground_truth):
    # MRR is equal to MAP when dataset is loo split.
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0 / (last_idx + 1)
    return result


def top_k_eval(ranks, ground_truths, k):
    hit_k_list = []
    precision_k_list = []
    recall_k_list = []
    map_k_list = []
    ndcg_k_list = []
    mrr_k_list = []
    ranks_k = [rank[:k] for rank in ranks]
    for i in range(0, len(ranks)):
        rank_i = ranks_k[i]
        hit_i_k = hit(rank_i, ground_truths[i])[-1]
        precision_i_k = precision(rank_i, ground_truths[i])[-1]
        recall_i_k = recall(rank_i, ground_truths[i])[-1]
        map_i_k = map(rank_i, ground_truths[i])[-1]
        ndcg_i_k = ndcg(rank_i, ground_truths[i])[-1]
        mrr_i_k = mrr(rank_i, ground_truths[i])[-1]
        hit_k_list.append(hit_i_k)
        precision_k_list.append(precision_i_k)
        recall_k_list.append(recall_i_k)
        map_k_list.append(map_i_k)
        ndcg_k_list.append(ndcg_i_k)
        mrr_k_list.append(mrr_i_k)
    hit_k = np.round(np.average(np.array(hit_k_list)), 4)
    precision_k = np.round(np.average(np.array(precision_k_list)), 4)
    recall_k = np.round(np.average(np.array(recall_k_list)), 4)
    map_k = np.round(np.average(np.array(map_k_list)), 4)
    ndcg_k = np.round(np.average(np.array(ndcg_k_list)), 4)
    mrr_k = np.round(np.average(np.array(mrr_k_list)), 4)
    return hit_k, precision_k, recall_k, map_k, ndcg_k, mrr_k

def rotation_mat2angle(R):
  return np.arccos(np.clamp((np.trace(R) - 1) / 2, -1, 1))


def rotation_error_old(R1, R2):
  assert R1.shape == R2.shape
  return np.arccos(np.clip((np.trace(np.dot(R1.T, R2)) - 1) / 2, -1, 1))

def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q

def matrix_from_quaternion(quaternion):
    theta=quaternion[0]
    w=theta
    x=quaternion[1]
    y=quaternion[2]
    z=quaternion[3]
    mat=np.eye(3)
    mat[0,0]=1-2*y*y-2*z*z
    mat[0,1]=2*x*y-2*z*w
    mat[0,2]=2*x*z+2*y*w
    mat[1,0]=2*x*y+2*z*w
    mat[1,1]=1-2*x*x-2*z*z
    mat[1,2]=2*y*z-2*x*w
    mat[2,0]=2*x*z-2*y*w
    mat[2,1]=2*y*z+2*x*w
    mat[2,2]=1-2*x*x-2*y*y
    return mat

def rotation_error(R_gt, R):
    eps = 1e-15
    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)
    return np.rad2deg(np.abs(err_q))

def translation_error(t1, t2):
  assert t1.shape == t2.shape
  return np.sqrt(((t1 - t2)**2).sum())


def batch_rotation_error(rots1, rots2):
  r"""
  arccos( (tr(R_1^T R_2) - 1) / 2 )
  rots1: B x 3 x 3 or B x 9
  rots1: B x 3 x 3 or B x 9
  """
  assert len(rots1) == len(rots2)
  trace_r1Tr2 = (rots1.reshape(-1, 9) * rots2.reshape(-1, 9)).sum(1)
  side = (trace_r1Tr2 - 1) / 2
  return np.arccos(np.clip(side, min=-1, max=1))


def batch_translation_error(trans1, trans2):
  r"""
  trans1: B x 3
  trans2: B x 3
  """
  assert len(trans1) == len(trans2)
  return np.norm(trans1 - trans2, p=2, dim=1, keepdim=False)


if __name__ == '__main__':

    translation1 = np.array([[0.515141 ,0.815774, 0.262951 ,-730.757874],
[-0.468592, 0.011173 ,0.883344, 898.350403],
[0.717671, -0.578263, 0.388021, 513.547852],
[0.000000, 0.000000, 0.000000 ,1.000000]])

    translation2 = np.array([[0.515141 ,0.815774, 0.262951 ,-730.757874],
[-0.468592, 0.011173 ,0.883344, 898.350403],
[0.717671, -0.578263, 0.388021, 523.547852],
[0.000000, 0.000000, 0.000000 ,1.000000]])

    print(rotation_error(translation1[:3,:3], translation2[:3,:3]))
    print(translation_error(translation1[:3,3], translation2[:3,3]))
