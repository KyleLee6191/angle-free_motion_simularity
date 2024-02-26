import numpy as np
import math
import torch
import matplotlib.pyplot as plt


def calculate_angle(v1, v2):
    cross = np.cross(v1, v2)
    cc = np.linalg.norm(cross)
    dd = np.dot(v1, v2)
    angle = math.atan2(cc, dd)
    angle = np.rad2deg(angle)
    if cross[2] < 0:
        angle *= -1
    return angle


# v1 = np.array((1,1,0))
# v2 = np.array((1,0,0))
#
#
# cross = np.cross(v1, v2)
# cc = np.linalg.norm(cross)
# dd = np.dot(v1, v2)
#
# angle_hip = math.atan2(cc, dd)
# angle_hip = np.rad2deg(angle_hip)
#
# print(calculate_angle(v1,v2))
# print(cross)


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)

def normalize_keypoints(keypoints):
    centroid = keypoints.mean(axis=1)[:, None]

    max_distance = np.max(np.sqrt(np.sum((keypoints - centroid) ** 2, axis=2)),
                          axis=1) + 1e-6

    normalized_keypoints = (keypoints - centroid) / max_distance[:, None, None]
    return normalized_keypoints

def object_keypoint_similarity(keypoints1,
                               keypoints2,
                               scale_constant=1.3,
                               keypoint_weights=None):
    """ Calculate the Object Keypoint Similarity (OKS) for multiple objects,
    and add weight to each keypoint. Here we choose to normalize the points
    using centroid and max distance instead of bounding box area.
    """

    # Compute squared distances between all pairs of keypoints
    keypoints1 = keypoints1.astype(np.float32)
    keypoints2 = keypoints2.astype(np.float32)

    sq_diff = np.sum((keypoints1[:, None] - keypoints2) ** 2, axis=-1)
    sq_diff = np.sqrt(sq_diff)

    oks = np.exp(-sq_diff / (2 * scale_constant ** 2))
    oks_unnorm = oks.copy()

    if keypoint_weights is not None:
        oks = oks * keypoint_weights
        oks = np.sum(oks, axis=-1)
    else:
        oks = np.mean(oks, axis=-1)

    return oks, oks_unnorm

def dynamic_time_warp(keypoints1, keypoints2, cost_matrix, R=2000):
    """Compute the Dynamic Time Warping distance and path between two time series.
    If the time series is too long, it will use the Sakoe-Chiba Band constraint,
    so time complexity is bounded at O(MR).
    """

    M = len(keypoints1)
    N = len(keypoints2)

    # Initialize the distance matrix with infinity
    D = np.full((M, N), np.inf)

    # Initialize the first row and column of the matrix
    D[0, 0] = cost_matrix[0, 0]
    for i in range(1, M):
        D[i, 0] = D[i - 1, 0] + cost_matrix[i, 0]

    for j in range(1, N):
        D[0, j] = D[0, j - 1] + cost_matrix[0, j]

    # Fill the remaining elements of the matrix within the
    # Sakoe-Chiba Band using dynamic programming
    for i in range(1, M):
        for j in range(max(1, i - R), min(N, i + R + 1)):
            cost = cost_matrix[i, j]
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtrack to find the optimal path
    path = [(M - 1, N - 1)]
    i, j = M - 1, N - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            min_idx = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
        elif i > 0 and j <= 0:
            min_idx = np.argmin([D[i - 1, 0], D[i, 0], D[i - 1, 0]])
        elif i <= 0 and j > 0:
            min_idx = np.argmin([D[0, j], D[0, j - 1], D[0, j - 1]])
        if min_idx == 0:
            i -= 1
        elif min_idx == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
        path.append((i, j))
    path.reverse()

    return D[-1, -1], path

def get_dtw_path(keypoints1, keypoints2):

    norm_kp1 = normalize_keypoints(keypoints1)
    norm_kp2 = normalize_keypoints(keypoints2)


    kp_weight = np.array((0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.3, 0.3, 0.4, 0.4, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0))
    kp_weight /= np.sum(kp_weight)

    oks, oks_unnorm = object_keypoint_similarity(norm_kp1,
                           norm_kp2, keypoint_weights=None)
    print(f"OKS max {oks.max():.2f} min {oks.min():.2f}")

    # do the DTW, and return the path
    cost_matrix = 1 - oks
    dtw_dist, dtw_path = dynamic_time_warp(keypoints1, keypoints2, cost_matrix)
    return dtw_path, oks, oks_unnorm

def get_path(keypoints1, keypoints2):


    point = keypoints1[0]
    point_2 = keypoints2[0]
    a1 = point[4, :] - point[1, :]
    a2 = point_2[4, :] - point_2[1, :]
    a1[2] = 0
    a2[2] = 0
    angle_hip = calculate_angle(a1,a2)
    print("angle_hip_first_frame = {}".format(angle_hip))
    c = math.cos(np.deg2rad(angle_hip))
    s = math.sin(np.deg2rad(angle_hip))
    transform = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    for i in range(len(keypoints2)):
        keypoints2[i] = [p @ transform for p in keypoints2[i]]
        keypoints2[i] = np.asarray(keypoints2[i])
    dtw_path, oks, oks_unnorm = get_dtw_path(keypoints1, keypoints2)
    return dtw_path, oks, oks_unnorm


def get_cos_simularity(dtw_path, keypoints1, keypoints2):
    angle_hip_mean = 0
    i = 0
    for pair in dtw_path:
        point = keypoints1[pair[0]]
        point_2 = keypoints2[pair[1]]
        a1 = point[4, :] - point[1, :]
        a2 = point_2[4, :] - point_2[1, :]
        a1[2] = 0
        a2[2] = 0
        angle_hip = calculate_angle(a1, a2)
        i += 1
        # print("angle_hip {} = {}".format(i, angle_hip))
        angle_hip_mean += angle_hip

    angle_hip_mean = angle_hip_mean / len(dtw_path)
    print("angle_hip_mean = {}".format(angle_hip_mean))
    c = math.cos(np.deg2rad(angle_hip_mean))
    s = math.sin(np.deg2rad(angle_hip_mean))
    transform = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    for i in range(len(keypoints2)):
        keypoints2[i] = [p @ transform for p in keypoints2[i]]
        keypoints2[i] = np.asarray(keypoints2[i])

    pairs = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
        (12, 13), (8, 14), (14, 15), (15, 16)
    ]

    directions3 = []
    for i in range(len(keypoints3)):
        directions = []
        for pair in pairs:
            direction_vector = (
                keypoints3[i][pair[1]][0] - keypoints3[i][pair[0]][0],
                keypoints3[i][pair[1]][1] - keypoints3[i][pair[0]][1],
                keypoints3[i][pair[1]][2] - keypoints3[i][pair[0]][2])

            # 计算方向向量的模长
            magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2 + direction_vector[2] ** 2)

            # 计算归一化方向向量
            normalized_direction_vector = (
                direction_vector[0] / magnitude,
                direction_vector[1] / magnitude,
                direction_vector[2] / magnitude
            )

            normalized_direction_vector = np.array(normalized_direction_vector)
            directions.append(normalized_direction_vector)
        directions = np.array(directions)
        directions3.append(directions)
    directions3 = np.array(directions3)

    directions4 = []
    for i in range(len(keypoints4)):
        directions = []
        for pair in pairs:
            direction_vector = (
                keypoints4[i][pair[1]][0] - keypoints4[i][pair[0]][0],
                keypoints4[i][pair[1]][1] - keypoints4[i][pair[0]][1],
                keypoints4[i][pair[1]][2] - keypoints4[i][pair[0]][2])

            # 计算方向向量的模长
            magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2 + direction_vector[2] ** 2)

            # 计算归一化方向向量
            normalized_direction_vector = (
                direction_vector[0] / magnitude,
                direction_vector[1] / magnitude,
                direction_vector[2] / magnitude
            )

            normalized_direction_vector = np.array(normalized_direction_vector)
            directions.append(normalized_direction_vector)
        directions = np.array(directions)
        directions4.append(directions)
    directions4 = np.array(directions4)


    angles_path = []
    for pair in dtw_path:
        angles = []


        for k in range(16):
            ang = np.cos(np.deg2rad(calculate_angle(directions3[pair[0]][k], directions4[pair[1]][k])))
            ang = 0 if ang < 0 else ang

            angles.append(ang)
        angles = np.array(angles)
        angles_path.append(angles)
    angles_path = np.array(angles_path)

    return angle_hip_mean, angles_path





if __name__ == '__main__':

    path1 = r"F:\PythonProjects\GraphMLP-main\demo\output\thby_ue_0\output_3D\output_keypoints_3d.npz"      # npz文件，3d关键点坐标
    path2 = r"F:\PythonProjects\GraphMLP-main\demo\output\thby_ue_45\output_3D\output_keypoints_3d.npz"

    points_3d = np.load(path1)
    keypoints1 = points_3d.f.reconstruction
    points_3d = np.load(path2)
    keypoints2 = points_3d.f.reconstruction


    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')

    for i in range(len(keypoints1)):
        keypoints1[i] = camera_to_world(keypoints1[i], R=rot, t=0)
        keypoints1[i][:, 2] -= np.min(keypoints1[i][:, 2])

    for i in range(len(keypoints2)):
        keypoints2[i] = camera_to_world(keypoints2[i], R=rot, t=0)
        keypoints2[i][:, 2] -= np.min(keypoints2[i][:, 2])

    keypoints3 = keypoints1.copy()
    keypoints4 = keypoints2.copy()
    dtw_path, oks, oke_unnorm = get_path(keypoints3,keypoints4)
    angle_hip, angles_path = get_cos_simularity(dtw_path, keypoints1, keypoints2)


    y = np.mean(angles_path, axis=-1)
    x = range(len(angles_path))
    plt.ylim(0, 1)
    plt.title("all")
    plt.xlabel('frame')
    plt.ylabel('oks')
    plt.plot(x, y)
    plt.show()

    plt.figure()
    kp_weight = np.array((0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.3, 0.3, 0.4, 0.4, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0))

    kp_weight /= np.sum(kp_weight)
    y = np.sum(angles_path * kp_weight, axis=-1)
    x = range(len(angles_path))
    plt.ylim(0, 1)
    plt.title("all_weighted")
    plt.xlabel('frame')
    plt.ylabel('oks')
    plt.plot(x, y)
    kp_weight = np.array((0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.3, 0.3, 0.4, 0.4, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0))
    kp_weight /= np.sum(kp_weight)
    plt.show()

    print("simularity = {}".format(np.mean(y)))

    plt.figure()
    plt.title("dtw_path")

    dtw_path = np.array(dtw_path)
    plt.plot(dtw_path[:, 0], dtw_path[:, 1])
    plt.show()
    print(111)

