import pandas as pd
import numpy as np
import math
from typing import List
from utils.features_encoding import encoding_features


class ToModelInput:
    def __init__(self, cfg):
        self.VELOCITY_THRESHOLD = cfg.velocity_threshold
        self.EXIST_THRESHOLD = cfg.exist_threshold
        self.model_column = ['timestamps', 'ID', 'Type', 'DX', 'DY', 'mask_traj']  # 目标轨迹保留的属性
        self.obs_len = cfg.obs_len           # 观测长度/历史信息长度
        self.norm_center = np.array([0, 0])  # 原点位置

    def main(self, all_prop_seg, lane_prop_seg=0):
        """
        Type:AV:主车-以其为坐标原点; Agent:主车的周围车辆 ---from argoverse1

        """
    # 1 数据整形，保留需要的属性
        # 1.1 目标轨迹数据整形
        all_prop_seg = all_prop_seg.rename(index=lambda x: x - 100)
        for column_name in all_prop_seg.columns:            # 保留需要的列名
            if column_name not in self.model_column:
                all_prop_seg = all_prop_seg.drop(column_name, axis=1)
        print('end')
        obs_prop = pd.DataFrame(columns=self.model_column)  # 按时间和ID进行排列
        for ID, df_piece in all_prop_seg.groupby('ID'):
            # 0.1s取点，不足的padding;timestamps重命名
            if df_piece.shape[0] < 100:
                df = pd.DataFrame(0, index=range(100), columns=self.model_column)
                index = df_piece.index
                df.loc[index, :] = df_piece.loc[index, :]
            else:
                df = df_piece.copy()
            df = df.loc[range(1, 100, 2)]
            df = df.rename(index=lambda x: int(x/2))
            df['timestamps'] = df.index/10
            obs_prop = pd.concat([obs_prop, df])
        obs_prop['Type'] = obs_prop['ID']                   # 赋予Type
        obs_prop.loc[obs_prop['Type'] != -1, 'Type'] = 'Agent'
        obs_prop.loc[obs_prop['Type'] == -1, 'Type'] = 'AV'
        columns_convert = ['timestamps', 'DX', 'DY']        # 类型转换为float
        obs_prop[columns_convert] = obs_prop[columns_convert].astype(float)
        obs_prop['mask_traj'] = 0
        # 1.2 车道线数据整形
        lane_feature_ls = 0



    # 2 格式转换-转成特征编码需要的输入格式
        # 2.1 障碍物数据格式转换
        AV_df = obs_prop.loc[obs_prop['Type'] == 'AV']
        dx, dy = AV_df[['DX', 'DY']].values[self.obs_len - 1]
        self.norm_center = np.array([dx, dy])
        obs_feature_all = self.get_nearby_moving_obj_feature_ls(obs_prop)
        AV_feature = self.get_AV_feature_ls(AV_df)
        # 2.2 车道线数据格式转换



        lane_feature_ls = self.get_nearby_lane_feature_ls(lane_prop_seg)



    # 3 特征编码-polyline-vectornet开源代码
        final_feature = encoding_features(AV_feature, obs_feature_all, lane_feature_ls)


        return final_feature, self.norm_center





    def get_nearby_lane_feature_ls(self, lane_line, norm_center, DeltaAngle, DeltaDx, DeltaDy):
        t = type(lane_line)
        halluc_lane_1 = [None] * 4
        halluc_lane_2 = [None] * 4
        lane_feature_ls = []

        L1_C0 = lane_line['L1_C0'].values[0]
        L1_C1 = lane_line['L1_C1'].values[0]
        L1_C2 = lane_line['L1_C2'].values[0]
        L1_C3 = lane_line['L1_C3'].values[0]
        L1_Start = lane_line['L1_Start'].values[0]
        L1_End = lane_line['L1_end'].values[0]

        R1_C0 = lane_line['R1_C0'].values[0]
        R1_C1 = lane_line['R1_C1'].values[0]
        R1_C2 = lane_line['R1_C2'].values[0]
        R1_C3 = lane_line['R1_C3'].values[0]
        R1_Start = lane_line['R1_Start'].values[0]
        R1_End = lane_line['R1_end'].values[0]
        if (L1_End != 0) & (R1_End != 0):
            lane_feature_ls = self.compute_lane(L1_C0, L1_C1, L1_C2, L1_C3, L1_Start, L1_End, R1_C0, R1_C1, R1_C2, R1_C3,
                                           R1_Start, R1_End, norm_center, lane_feature_ls, DeltaAngle, DeltaDx, DeltaDy)

        L2_C0 = lane_line['L2_C0'].values[0]
        L2_C1 = lane_line['L2_C1'].values[0]
        L2_C2 = lane_line['L2_C2'].values[0]
        L2_C3 = lane_line['L2_C3'].values[0]
        L2_Start = lane_line['L2_Start'].values[0]
        L2_End = lane_line['L2_end'].values[0]

        L3_C0 = lane_line['L3_C0'].values[0]
        L3_C1 = lane_line['L3_C1'].values[0]
        L3_C2 = lane_line['L3_C2'].values[0]
        L3_C3 = lane_line['L3_C3'].values[0]
        L3_Start = lane_line['L3_Start'].values[0]
        L3_End = lane_line['L3_end'].values[0]

        if (L2_End != 0) & (L3_End != 0):
            lane_feature_ls = self.compute_lane(L2_C0, L2_C1, L2_C2, L2_C3, L2_Start,
                                           L2_End, L3_C0, L3_C1, L3_C2, L3_C3, L3_Start,
                                           L3_End, norm_center, lane_feature_ls, DeltaAngle, DeltaDx, DeltaDy)

        R2_C0 = lane_line['R2_C0'].values[0]
        R2_C1 = lane_line['R2_C1'].values[0]
        R2_C2 = lane_line['R2_C2'].values[0]
        R2_C3 = lane_line['R2_C3'].values[0]
        R2_Start = lane_line['R2_Start'].values[0]
        R2_End = lane_line['R2_end'].values[0]

        R3_C0 = lane_line['R3_C0'].values[0]
        R3_C1 = lane_line['R3_C1'].values[0]
        R3_C2 = lane_line['R3_C2'].values[0]
        R3_C3 = lane_line['R3_C3'].values[0]
        R3_Start = lane_line['R3_Start'].values[0]
        R3_End = lane_line['R3_end'].values[0]

        if (R2_End != 0) & (R3_End != 0):
            lane_feature_ls = self.compute_lane(R2_C0, R2_C1, R2_C2, R2_C3, R2_Start, R2_End,
                                           R3_C0, R3_C1, R3_C2, R3_C3, R3_Start, R3_End, norm_center, lane_feature_ls,
                                           DeltaAngle, DeltaDx, DeltaDy)

        return lane_feature_ls

    def compute_lane(C0_1, C1_1, C2_1, C3_1, Start_1, End_1, C0_2, C1_2, C2_2, C3_2, Start_2, End_2, norm_center,
                     lane_feature_ls, DeltaAngle, DeltaDx, DeltaDy):
        if (C2_1 <= 0.0004) & (C2_2 <= 0.0004):
            if End_1 - Start_1 <= 100:
                n = 5
            elif End_1 - Start_1 <= 200:
                n = 10
            else:
                n = 15
            num = (End_1 - Start_1) / n
            line_all = np.arange(Start_1, End_1 + num, num)
        elif C2_1 >= C2_2:
            if End_1 - Start_1 <= 100:
                n = (C2_1 / 0.004 + 1) * 5
            elif End_1 - Start_1 <= 200:
                n = (C2_1 / 0.004 + 1) * 10
            else:
                n = (C2_1 / 0.004 + 1) * 15
            num = (End_1 - Start_1) / n
            line_all = np.arange(Start_1, End_1 + num, num)
        elif C2_1 < C2_2:
            if End_1 - Start_1 <= 100:
                n = (C2_2 / 0.004 + 1) * 5
            elif End_1 - Start_1 <= 200:
                n = (C2_2 / 0.004 + 1) * 10
            else:
                n = (C2_2 / 0.004 + 1) * 15
            num = (End_2 - Start_2) / n
            line_all = np.arange(Start_2, End_2 + num, num)

        h = 0
        for i in range(1, len(line_all)):
            halluc_lane_1, halluc_lane_2 = [None] * 4, [None] * 4
            h = h + 1
            # print('这是第' + str(h) +'轮')
            step = (line_all[i] - line_all[i - 1]) / 10
            li_x = np.arange(line_all[i - 1], line_all[i], step)
            norm_center = np.float32(norm_center)
            y1_1 = len(li_x)
            for li in range(1, len(li_x)):
                # print(li)
                y1_1 = C1_1 * li_x[li - 1] + C2_1 * li_x[li - 1] ** 2 + C3_1 * li_x[li - 1] ** 3 + C0_1
                y1_2 = C1_1 * li_x[li] + C2_1 * li_x[li] ** 2 + C3_1 * li_x[li] ** 3 + C0_1
                y2_1 = C1_2 * li_x[li - 1] + C2_2 * li_x[li - 1] ** 2 + C3_2 * li_x[li - 1] ** 3 + C0_2
                y2_2 = C1_2 * li_x[li] + C2_2 * li_x[li] ** 2 + C3_2 * li_x[li] ** 3 + C0_2
                x1_1 = li_x[li - 1]
                x1_2 = li_x[li]
                x2_1 = li_x[li - 1]
                x2_2 = li_x[li]

                for lane_ch in range(0, len(DeltaAngle)):
                    x1_1 = x1_1 - DeltaDx[lane_ch]
                    y1_1 = y1_1 - DeltaDy[lane_ch]
                    x1_1 = (x1_1 * (math.cos(DeltaAngle[lane_ch]))) + (y1_1 * (math.sin(DeltaAngle[lane_ch])))
                    y1_1 = (-x1_1 * (math.sin(DeltaAngle[lane_ch]))) + (y1_1 * (math.cos(DeltaAngle[lane_ch])))

                    x1_2 = x1_2 - DeltaDx[lane_ch]
                    y1_2 = y1_2 - DeltaDy[lane_ch]
                    x1_2 = (x1_2 * (math.cos(DeltaAngle[lane_ch]))) + (y1_2 * (math.sin(DeltaAngle[lane_ch])))
                    y1_2 = (-x1_2 * (math.sin(DeltaAngle[lane_ch]))) + (y1_2 * (math.cos(DeltaAngle[lane_ch])))

                    x2_1 = x2_1 - DeltaDx[lane_ch]
                    y2_1 = y2_1 - DeltaDy[lane_ch]
                    x2_1 = (x2_1 * (math.cos(DeltaAngle[lane_ch]))) + (y2_1 * (math.sin(DeltaAngle[lane_ch])))
                    y2_1 = (-x2_1 * (math.sin(DeltaAngle[lane_ch]))) + (y2_1 * (math.cos(DeltaAngle[lane_ch])))

                    x2_2 = x2_2 - DeltaDx[lane_ch]
                    y2_2 = y2_2 - DeltaDy[lane_ch]
                    x2_2 = (x2_2 * (math.cos(DeltaAngle[lane_ch]))) + (y2_2 * (math.sin(DeltaAngle[lane_ch])))
                    y2_2 = (-x2_2 * (math.sin(DeltaAngle[lane_ch]))) + (y2_2 * (math.cos(DeltaAngle[lane_ch])))

                halluc_lane_1 = np.vstack((halluc_lane_1,
                                           [x1_1 - norm_center[0], y1_1 - norm_center[1], x1_2 - norm_center[0],
                                            y1_2 - norm_center[1]]))
                halluc_lane_2 = np.vstack((halluc_lane_2,
                                           [x2_1 - norm_center[0], y2_1 - norm_center[1], x2_2 - norm_center[0],
                                            y2_2 - norm_center[1]]))
            lane_feature_ls.append([halluc_lane_1[1:], halluc_lane_2[1:]])
        return lane_feature_ls





# ################################ 未动工具函数 #########################################
    def get_AV_feature_ls(self, AV_df):
        """
        args:
        returns:
            list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
        """
        xys, gt_xys = AV_df[["DX", "DY"]].values[:self.obs_len], AV_df[["DX", "DY"]].values[self.obs_len:]
        xys -= self.norm_center  # normalize to last observed timestamp point of agent
        xys = np.hstack((xys[:-1], xys[1:]))

        ts = AV_df['time'].values[:self.obs_len]
        ts = (ts[:-1] + ts[1:]) / 2
        real_xy = AV_df[['DX', 'DY']].values[self.obs_len:] - self.norm_center
        real_xy_mask = AV_df[['mask_traj']].values[self.obs_len:]

        return [xys, AV_df['Type'].values[0], ts, AV_df['ID'].values[0], AV_df['mask_traj'].values, real_xy,
                real_xy_mask]

    def get_nearby_moving_obj_feature_ls(self, group_traj):
        obj_feature_ls = []

        for track_id, remain_df in group_traj.groupby('ID'):
            if remain_df['Type'].values[0] == 'AV':
                continue

            if len(remain_df) < self.EXIST_THRESHOLD or self.get_is_track_stationary(remain_df):
                continue

            xys = remain_df[['DX', 'DY']].values[: self.obs_len]
            ts = remain_df["timestamps"].values[: self.obs_len]

            xys -= self.norm_center  # normalize to last observed timestamp point of agent
            xys = np.hstack((xys[:-1], xys[1:]))

            ts = (ts[:-1] + ts[1:]) / 2
            real_xy = remain_df[['DX', 'DY']].values[self.obs_len:] - self.norm_center - xys[-1, 2:]
            real_xy_mask = remain_df[['mask_traj']].values[self.obs_len:]

            obj_feature_ls.append(
                [xys, remain_df['Type'].values[0], ts, track_id, remain_df['mask_traj'].values[: self.obs_len], real_xy,
                 real_xy_mask])
        return obj_feature_ls

    def get_is_track_stationary(self, track_df: pd.DataFrame) -> bool:
        """Check if the track is stationary.

        Args:
            track_df (pandas Dataframe): Data for the track
        Return:
            _ (bool): True if track is stationary, else False

        """
        vel = self.compute_velocity(track_df)
        sorted_vel = sorted(vel)
        threshold_vel = sorted_vel[int(len(vel) / 2)]
        return True if threshold_vel < self.VELOCITY_THRESHOLD else False

    def compute_velocity(self, track_df: pd.DataFrame) -> List[float]:
        """Compute velocities for the given track.

        Args:
            track_df (pandas Dataframe): Data for the track
        Returns:
            vel (list of float): Velocity at each timestep

        """
        x_coord = track_df["DX"].values
        y_coord = track_df["DY"].values
        timestamp = track_df["timestamps"].values
        vel_x, vel_y = zip(*[(
            float(x_coord[i]) - float(x_coord[i - 1]) /
            (float(timestamp[i]) - float(timestamp[i - 1])),
            float(y_coord[i]) - float(y_coord[i - 1]) /
            (float(timestamp[i]) - float(timestamp[i - 1])),
        ) for i in range(1, len(timestamp))])
        vel = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(vel_x, vel_y)]

        return vel

