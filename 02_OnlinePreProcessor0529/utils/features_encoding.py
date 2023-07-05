import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.viz_utils import *
import warnings
import pdb

def encoding_features(AV_feature, obj_feature_ls, lane_feature_ls):
    """
        args:
            agent_feature_ls:
                list of (doubeld_track, object_type, timestamp, track_id, not_doubled_groudtruth_feature_trajectory)
            obj_feature_ls:
                list of list of (doubled_track, object_type, timestamp, track_id)
            lane_feature_ls:
                list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
        returns:
            pd.DataFrame of (
                polyline_features: vstack[
                    (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id),
                    (xs, ys, xe, ye, NULL, zs, ze, polyline_id)
                    ]
                offset_gt: incremental offset from agent's last obseved point,
                traj_id2mask: Dict[int, int]
                lane_id2mask: Dict[int, int]
            )
            where obejct_type = {0 - others, 1 - agent}

        """
    polyline_id = 0
    traj_id2mask, lane_id2mask = {}, {}
    traj_nd, lane_nd = np.empty((0, 7)), np.empty((0, 7))
    traj_nd_real = np.empty((0, 2))
    traj_real_mask = np.empty((0, 1))
    # encoding agent feature
    pre_traj_len = traj_nd.shape[0]
    AV_len = AV_feature[0].shape[0]
    AV_nd = np.hstack((AV_feature[0], np.ones(
        (AV_len, 1)), AV_feature[2].reshape((-1, 1)), np.ones((AV_len, 1)) * polyline_id))
    assert AV_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

    traj_nd = np.vstack((AV_nd, traj_nd))
    av_nd_real = AV_feature[-2]
    av_nd_mask = AV_feature[-1]
    traj_nd_real = np.vstack((traj_nd_real, av_nd_real))
    traj_real_mask = np.vstack((traj_real_mask, av_nd_mask))

    traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
    pre_traj_len = traj_nd.shape[0]
    polyline_id += 1

    # encoding obj feature
    for obj_feature in obj_feature_ls:
        obj_len = obj_feature[0].shape[0]
        # assert obj_feature[2].shape[0] == obj_len, f"obs_len of obj is {obj_len}"
        if not obj_feature[2].shape[0] == obj_len:
            from pdb import set_trace;
            set_trace()
        obj_nd = np.hstack((obj_feature[0], np.zeros(
            (obj_len, 1)), obj_feature[2].reshape((-1, 1)), np.ones((obj_len, 1)) * polyline_id))
        assert obj_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        traj_nd = np.vstack((traj_nd, obj_nd))
        obj_nd_real = obj_feature[-2]
        obj_nd_mask = obj_feature[-1]
        traj_nd_real = np.vstack((traj_nd_real, obj_nd_real))
        traj_real_mask = np.vstack((traj_real_mask, obj_nd_mask))
        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1

    # incodeing lane feature
    pre_lane_len = lane_nd.shape[0]
    for lane_feature in lane_feature_ls:
        l_lane_len = lane_feature[0].shape[0]
        l_lane_nd = np.hstack(
            (lane_feature[0], np.zeros((l_lane_len, 2)), np.ones((l_lane_len, 1)) * polyline_id))
        assert l_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, l_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_1 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        r_lane_len = lane_feature[1].shape[0]
        r_lane_nd = np.hstack(
            (lane_feature[1], np.zeros((r_lane_len, 2)), np.ones((r_lane_len, 1)) * polyline_id)
        )
        assert r_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, r_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_2 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        assert _tmp_len_1 == _tmp_len_2, f"left, right lane vector length contradict"
        # lane_nd = np.vstack((lane_nd, l_lane_nd, r_lane_nd))

        # FIXME: handling `nan` in lane_nd
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_mean = np.nanmean(lane_nd, axis=0)

    offset_obj = trans_obj_offset_format(traj_nd_real)

    lane_nd = np.hstack(
        [lane_nd, np.zeros((lane_nd.shape[0], 1), dtype=lane_nd.dtype)])
    lane_nd = lane_nd[:, [0, 1, 2, 3, 7, 4, 5, 6]]
    # change object features to (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id)
    traj_nd = np.hstack(
        [traj_nd, np.zeros((traj_nd.shape[0], 2), dtype=traj_nd.dtype)])
    traj_nd = traj_nd[:, [0, 1, 2, 3, 5, 7, 8, 6]]

    # don't ignore the id
    polyline_features = np.vstack((traj_nd, lane_nd))
    np.set_printoptions(threshold=np.inf)
    data = [[polyline_features.astype(
        np.float32), offset_obj, traj_id2mask, lane_id2mask, traj_nd.shape[0], lane_nd.shape[0], traj_real_mask]]

    return pd.DataFrame(
        data,
        columns=["POLYLINE_FEATURES", "Offset_obj",
                 "TRAJ_ID_TO_MASK", "LANE_ID_TO_MASK", "TRAJ_LEN", "LANE_LEN", 'Real_mask']
    )

def save_features(df, name, dir_=None):
    if dir_ is None:
        dir_ = '/mnt/hgfs/07data/ours_data/test_interm/bicycle_test/ele_test_intermediate'
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    name = f"features_{name}.pkl"
    df.to_pickle(
        os.path.join(dir_, name)
    )


def trans_obj_offset_format(traj_nd_real):
    """
    >Our predicted trajectories are parameterized as per-stepcoordinate offsets, starting from the last observed location.We rotate the coordinate system based on the heading of the target vehicle at the last observed location.

    """
    if traj_nd_real.shape == (0, 2):
        return traj_nd_real
    len_obj_traj = traj_nd_real.shape[0]
    offset_obj = np.empty((0, 2))
    for i in range(0, len_obj_traj, 30):
        offset_obj_nd = np.vstack((traj_nd_real[i], traj_nd_real[i+1:i+30] - traj_nd_real[i:i+29]))
        offset_obj = np.vstack((offset_obj, offset_obj_nd))
        assert (offset_obj[i:i+30].cumsum(axis=0) -
                traj_nd_real[i:i+30]).sum() < 1e-6, f"{(offset_obj[i:i+30].cumsum(axis=0) - traj_nd_real[i:i+30]).sum()} "

    return offset_obj
