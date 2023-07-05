from dataclasses import dataclass
from utils.config import DefaultConfig
from utils.mdf_reader import Mf4Reader
import pandas as pd
import numpy as np
from utils.channelist import *
from utils.utils import sinc, poly_func
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class TrajPreprocessor:
    cfg: DefaultConfig
    reader: Mf4Reader

    def get_adc_info(self):
        """
        get ADC information from MF4
        """
        adc_prop_mid = self.reader.get_dataFrame(EMP_self_property())
        adc_prop_mid = adc_prop_mid.reset_index()  # 读取数据然后reset_index   后续都是如此
        adc_prop = pd.DataFrame(columns=cols_Refms)
        adc_prop = adc_prop.append(adc_prop_mid)
        adc_prop['ID'].iloc[:] = -1  # 新增ID列，自车ID=-1
        return adc_prop

    @staticmethod
    def get_transform_matrix(adc_prop_w):
        adc_prop = adc_prop_w
        dt = 0.05
        # 每个时刻的速度和角速度
        adc_vx = adc_prop['EMP_vxvRefMs'].values
        adc_w = adc_prop['EMP_psiDtOpt'].values
        # 基于当前时刻与上一时刻，计算平均速度和平均角速度
        adc_vxm = (adc_vx[:-1] + adc_vx[1:])/2.
        adc_wm = (adc_w[:-1] + adc_w[1:])/2.
        adc_anglem = adc_wm * dt
        taylor_poly = np.asarray([sinc(angle) for angle in adc_anglem])
        half_taylor_poly = np.asarray([sinc(angle/2.) for angle in adc_anglem])
        # 求解平移矩阵
        delta_dx = adc_vxm * taylor_poly * dt
        delta_dy = adc_vxm * np.sin(adc_anglem/2.) * half_taylor_poly * dt
        # 求解旋转矩阵，坐标轴变化后，逆时针旋转变为顺时针旋转，平移矩阵由正变负
        rot_mat = np.asarray([[np.cos(adc_anglem), np.sin(adc_anglem)],
                              [-np.sin(adc_anglem), np.cos(adc_anglem)]]).transpose([2, 0, 1])
        trans = np.asarray([-delta_dx, -delta_dy]).T

        return rot_mat, trans

    @staticmethod
    # 坐标转换转到最后一帧为原点
    def get_transform_trajectory(all_prop_w, adc_prop_w):
        all_prop = all_prop_w
        rot_mat, trans = TrajPreprocessor.get_transform_matrix(adc_prop_w)
        timestamps = np.sort(all_prop['timestamps'].unique())
        total_num = len(timestamps)
        all_prop_trans = all_prop.copy(deep=True)
        assert (len(timestamps) == len(rot_mat) + 1)
        for index, ts in enumerate(timestamps[:-1]):
            if index > total_num:
                break
            x_orig = all_prop_trans.loc[all_prop_trans['timestamps'] == ts]['DX']
            y_orig = all_prop_trans.loc[all_prop_trans['timestamps'] == ts]['DY']
            x_trans, y_trans = x_orig, y_orig
            for rm, tr in zip(rot_mat[index:total_num], trans[index:total_num]):
                x_trans, y_trans = x_trans + tr[0], y_trans + tr[1]
                x_trans = x_trans * rm[0][0] + y_trans * rm[0][1]
                y_trans = x_trans * rm[1][0] + y_trans * rm[1][1]
            all_prop_trans.loc[all_prop_trans['timestamps'] == ts, 'DX'] = x_trans
            all_prop_trans.loc[all_prop_trans['timestamps'] == ts, 'DY'] = y_trans
        return all_prop_trans

    def get_obj_info(self):
        """
        get object information from MF4
        """
        # 获得160个目标属性信息： obj_160_property
        _160_tal_others = pd.DataFrame(columns=cols_obj_160)
        for i in range(0, 160):
            obj_160_prop = self.reader.get_dataFrame(OFM_obj_160(i))
            obj_160_prop = obj_160_prop.reset_index()
            obj_160_prop = obj_rename(obj_160_prop, i)  # 将mf4记录的复杂名字，换成便于理解的
            obj_160_prop = obj_160_prop.loc[obj_160_prop['ID'] != 0]  # 全时间戳，将ID不存在的行删除，保留有数据的时间戳
            _160_tal_others = _160_tal_others.append(obj_160_prop)
        _160_tal_others = pd.DataFrame(_160_tal_others, columns=cols_obj_160)  # 重塑列
        _160_tal_others.sort_values(['timestamps'], ascending=True, inplace=True)
        # 对所有目标（obj_160和self）整合，时间戳排序
        adc_prop = self.get_adc_info()
        all_prop = _160_tal_others.append(adc_prop)
        # 自车的[DX, DY] 赋值 [0, 0]
        all_prop.loc[all_prop['ID'] == -1, 'DX'] = 0.
        all_prop.loc[all_prop['ID'] == -1, 'DY'] = 0.
        all_prop.sort_values(['timestamps'], ascending=True, inplace=True)
        return all_prop

    def obj_screen(self):
        all_prop = self.get_obj_info()
        timestamps = np.sort(all_prop['timestamps'].unique())
        all_prop_aft = pd.DataFrame([], columns=all_prop.columns)
        for index, ts in enumerate(timestamps):
            data_piece = all_prop.loc[all_prop['timestamps'] == ts]
            data_piece_aft = data_piece
            for ID, obj_df in data_piece.groupby('ID'):  # cur_IDdf 当前时刻目标ID的信息
                if ID == -1:
                    continue
                # 筛选规则 当前时刻自车前方100m内 左右-10~10m 超出范围的删除
                if (obj_df['DX'].iloc[0] > self.cfg.max_dx) or (abs(obj_df['DY'].iloc[0]) > self.cfg.max_dy):
                    data_piece_aft = data_piece_aft[data_piece_aft.ID != ID]  # 删除ID的所有行
                    continue
                # 筛选规则 目标age<60 (少于3s) 删除
                if obj_df['Age'].iloc[0] < 20:
                    data_piece_aft = data_piece_aft[data_piece_aft.ID != ID]  # 删除ID的所有行
                    continue
                # 筛选规则 目标age<100 且融合类型不是车辆 删除
                if obj_df['Age'].iloc[0] < 60 and (
                        obj_df['Type'].iloc[0] != 2 or obj_df['Type'].iloc[0] != 3):
                    data_piece_aft = data_piece_aft[data_piece_aft.ID != ID]  # 删除ID的所有行
                    continue
                # 筛选规则 目标单雷达目标 且运动状态为静止 删除
                if obj_df['Fus_Type'].iloc[0] == 2 and (
                        obj_df['MovingState'].iloc[0] != 2 or obj_df['MovingState'].iloc[0] != 4 or
                        obj_df['MovingState'].iloc[0] != 5):
                    data_piece_aft = data_piece_aft[data_piece_aft.ID != ID]  # 删除ID的所有行
                    continue
            all_prop_aft = all_prop_aft.append(data_piece_aft)
        return all_prop_aft

    def slide_window(self):
        """
        生成目标滑动窗口
        """
        # 滑动窗口参数
        window_size = self.cfg.window_size  # 时间窗口为16秒（320帧）
        distance_window = self.cfg.distance_window   # 自车行驶距离为50米
        forward = self.cfg.forward  # 下一个窗口向前推进的距离
        time_col = 'timestamps'
        #
        data = self.get_adc_info()
        data = data.sort_values(time_col)  # 按照时间戳排序
        windows = []
        if len(data) < window_size:
            windows.append([0, len[data] - 1])
            return np.array(windows)
        # 窗口起始和结束的索引
        start_index = 0
        end_index = window_size - 1
        while start_index < (len(data) - 1) and end_index < (len(data) - 1):
            windows.append([start_index, end_index])
            # 提取窗口内的数据
            window_data = data.iloc[start_index:end_index]
            window_data = window_data[window_data['ID'] == -1]
            time_diff = window_data['timestamps'].diff()
            # 删除第一行
            time_diff = time_diff.iloc[1:]
            # 将索引加1
            time_diff.index -= 1
            dx = window_data['EMP_vxvRefMs'] * time_diff + window_data['EMP_axRefMs2'] ** 2 * time_diff * 0.5
            dy = window_data['EMP_ayvRefMs2'] * (time_diff ** 2) * 0.5
            distance = np.sqrt(dx + dy)
            total_distance = distance.sum()

            if (total_distance > distance_window):
                start_index += forward
                end_index += forward
            else:
                start_index += forward
                end_index = start_index + window_size*2 - 1
        if end_index > len(data):
            windows.append([start_index, len(data) - 1])
        return np.array(windows)

    @staticmethod
    def curve_fitting(all_prop_trans, fp=8):
        """
        区间内曲线拟合
        """
        all_prop_ed = all_prop_trans.copy()
        # 按照ID进行分组
        grouped_all_prop_w = all_prop_ed.groupby('ID')
        # 遍历分组数据
        for obj_ID, group_data in grouped_all_prop_w:
            X = np.array(group_data['timestamps'])
            y_dx = np.array(group_data['DX'])
            y_dy = np.array(group_data['DY'])
            # 根据nh参数设置多项式系数的个数
            if len(group_data) > fp:
                n_coeff = fp + 1
            else:
                continue
            # 多项式系数
            init_coeff = np.zeros(n_coeff)
            popt_dx, pcov_dx = curve_fit(poly_func, X, y_dx, p0=init_coeff, maxfev=20000)
            popt_dy, pcov_dy = curve_fit(poly_func, X, y_dy, p0=init_coeff, maxfev=20000)
            # 数据修正
            # todo:判断离群点，离群点朝曲线收；目前是将曲线值当作修正后结果
            p_dy = poly_func(X, *popt_dy)
            p_dx = poly_func(X, *popt_dx)
            # 数据回落
            ID = group_data.iloc[0]['ID']
            data = all_prop_ed[all_prop_ed['ID'] == ID]
            data['DX'] = p_dx
            data['DY'] = p_dy
            all_prop_ed[all_prop_ed['ID'] == ID] = data
        return all_prop_ed

