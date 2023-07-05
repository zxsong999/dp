
import matplotlib.pyplot as plt
import numpy as np
import torch

color_dict = {"AGENT": "red", "OTHERS": "purple", "AV": "#007672"}
def show_doubled_lane(polygon):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """

    xs, ys = polygon[:, 0], polygon[:, 1]
    plt.plot(xs, ys, '--', color='grey')


def show_traj(traj, type_):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    plt.ylim(-20, 20)
    plt.xlim(-150, 150)
    plt.plot(traj[:, 0], traj[:, 1], color=color_dict[type_])


def reconstract_polyline(features, traj_mask, lane_mask, add_len):
    traj_ls, lane_ls = [], []
    for id_, mask in traj_mask.items():
        data = features[mask[0]: mask[1]]
        traj = np.vstack((data[:, 0:2], data[-1, 2:4]))
        traj_ls.append(traj)
    for id_, mask in lane_mask.items():
        data = features[mask[0]+add_len: mask[1]+add_len]
        # lane = np.vstack((data[:, 0:2], data[-1, 3:5]))
        # change lanes feature to (xs, ys, zs, xe, ye, ze, polyline_id)
        lane = np.vstack((data[:, 0:2], data[-1, 2:4]))
        lane_ls.append(lane)
    return traj_ls, lane_ls


def show_pred_and_gt(pred_y, y):

    plt.plot(y[:, 0], y[:, 1], color='yellow')
    plt.plot(pred_y[:, 0], pred_y[:, 1], lw=0, marker='o', fillstyle='none', color='pink')


def show_predict_result(norm_centers_ls, data, pred_obj_y: torch.Tensor, obj_y: torch.Tensor, add_len, show_lane=True, ):
    features, _ = data['POLYLINE_FEATURES'].values[0], data['Offset_obj'].values[0].astype(
        np.float32)

    traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]

    traj_ls, lane_ls = reconstract_polyline(
        features, traj_mask, lane_mask, add_len)

    type_ = 'AV'
    for traj in traj_ls:
        # for i in range(20):
            # traj[i, 0] = traj[i, 0] + norm_centers_ls[0]
            # traj[i, 1] = traj[i, 1] + norm_centers_ls[1]
        show_traj(traj, type_)
        type_ = 'AGENT'

    if show_lane:
        for lane in lane_ls:
            # for i in range(10):
            #     lane[i, 0] = lane[i, 0] + norm_centers_ls[0]
            #     lane[i, 1] = lane[i, 1] + norm_centers_ls[1]
            show_doubled_lane(lane)
    # pred_y[0, 0] = pred_y[0, 0] + norm_centers_ls[0]
    # pred_y[0, 1] = pred_y[0, 1] + norm_centers_ls[1]
    # y[0, 0] = y[0, 0] + norm_centers_ls[0]
    # y[0, 1] = y[0, 1] + norm_centers_ls[1]
    len_obj = pred_obj_y.shape[0]
    j = 1
    for i in range(0, len_obj, 60):
        pred_y = pred_obj_y[i:i+60]
        pred_y = (pred_y.numpy().reshape((-1, 2)).cumsum(axis=0)) + features[(j * 19 - 1), 2:4]
        y = (obj_y[i:i+60].numpy().reshape((-1, 2)).cumsum(axis=0)) + features[(j * 19 - 1), 2:4]
        j+=1
        show_pred_and_gt(pred_y, y)
