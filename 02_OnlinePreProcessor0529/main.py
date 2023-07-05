from traj_preprocessor import TrajPreprocessor
from lane_preprocessor import LaneProcessor
from utils.channelist import lane_RFM_property
from utils.mdf_reader import Mf4Reader
from utils.visualize import ed_visualize
from to_model_input import ToModelInput
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from utils.config import DefaultConfig
import os
from utils.features_encoding import save_features
import warnings
import pickle
warnings.filterwarnings("ignore")

# 数据处理主函数
def process_core(file_start, file_end):
    cfg = DefaultConfig
    # 文件操作
    files_list = os.listdir(cfg.file_path)
    for file in tqdm(files_list[int(file_start): int(file_end)]):
        file_type = str(file).split('.')[-1]  # 读取文件格式，如excel还是mf4等
        if file_type != cfg.tar_type:  # 如果文件格式不相同，则跳过该文件
            continue
        path_dir = cfg.file_path + '/' + file  # 获取目标文件格式在源文件的路径
        reader = Mf4Reader("{}".format(path_dir))  # 读取mf4文件
        traj_pp = TrajPreprocessor(cfg=cfg, reader=reader)
        lane_pp = LaneProcessor(cfg=cfg, reader=reader, lane_property=lane_RFM_property())

        # 模拟当前时刻，前端数据获取
        all_prop = traj_pp.obj_screen()
        lane_prop = lane_pp.lane_screen()
        adc_prop = traj_pp.get_adc_info()
        # 当前时刻数据存储buffer
        slide_win = traj_pp.slide_window()
        for index, w in enumerate(slide_win):        
            if index<77:
                continue

            all_prop_w = all_prop.loc[w[0]:w[1]]
            lane_prop_w = lane_prop.loc[w[0]:w[1]]
            adc_prop_w = adc_prop.loc[w[0]:w[1]]


# ——————————————在线数据前处理——————————————————————————————————————————————————————————————————————————————————————————
            # 1 目标轨迹：tracking、坐标转换、数据修正-曲线拟合
            # TODO all_prop_w = TrajPreprocessor.tracking(all_prop_w)
            all_prop_trans = TrajPreprocessor.get_transform_trajectory(all_prop_w, adc_prop_w)
            all_prop_ed = traj_pp.curve_fitting(all_prop_trans)

            # 2 车道线：tracking、坐标转换、数据修正-曲线拟合；车道线三段表示
            lane_prop_w, split_dataframes = LaneProcessor.tracking(lane_prop_w, all_prop_trans.loc[all_prop_trans['ID'] == -1])
            lane_prop_trans = lane_pp.get_transform_lane(lane_prop_w, adc_prop_w)
            lane_prop_ed = LaneProcessor.curve_fitting(lane_prop_trans)

            # 3 画图
            if index < 10:
                name = f'{file[20:-4]}_{w[0]}-{w[1]}=0{index}'
            else:
                name = f'{file[20:-4]}_{w[0]}-{w[1]}={index}'
            ed_visualize(all_prop_trans, lane_prop_trans, all_prop_ed, lane_prop_ed, name)
            print(index)

    # # 3 数据结构转模型输入
    # # 3.1 数据切片 坐标转换
    #         # 数据切片  16s轨迹切片 6-10s
    #         seg = np.array([100, 199])
    #         all_prop_seg = all_prop_ed.loc[seg[0]:seg[1]]
    #         # lane_prop_seg = lane_prop_ed.loc[seg[0]:seg[1]]
    #         # 坐标转换
    #         adc_prop_seg = all_prop_seg.loc[all_prop_seg['ID'] == -1]
    #         adc_dx, adc_dy = adc_prop_seg.loc[seg[1], 'DX'], adc_prop_seg.loc[seg[1], 'DY']
    #         all_prop_seg['DX'], all_prop_seg['DY'] = all_prop_seg['DX'] - adc_dx, all_prop_seg['DY'] - adc_dy
    #         # lane_prop_seg = TrajPreprocessor.get_transform_trajectory(lane_prop_seg, adc_prop_seg)

    # # 3.2 得到pkl
    #         to_model_input = ToModelInput(cfg)
    #         final_feature, norm_center = to_model_input.main(all_prop_seg)

    # # 3.3 存储
    #         pkl_path = cfg.pkl_path
    #         save_features(final_feature, name, os.path.join(
    #             pkl_path, f"1223test_intermediate"))

    #         # check保存结果
    #         norm_center_dict = {name: norm_center}
    #         with open(os.path.join(pkl_path, f"1223test_intermediate.pkl"), 'wb') as f:
    #             pickle.dump(norm_center_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":  # 读取文件夹mf4文件，并转成excel格式，
    cfg = DefaultConfig()
    # 路径
    file_path = cfg.file_path
    save_path = cfg.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filesList = os.listdir(file_path)
    # 并行与否
    if cfg.parallel_compute == 'off':
        file_start, file_end = 0, len(filesList)
        process_core(file_start, file_end)
        print('finish all')

    if cfg.parallel_compute == 'on':
        pool_num = cfg.pool_num
        start_num, end_num = np.zeros(pool_num), np.zeros(pool_num)  # 为每个进程分配文件
        pool = Pool(pool_num)  # 定义一个进程池，最大进程数为30
        for i in range(pool_num):
            list_num = len(filesList) // pool_num
            list_re = len(filesList) % pool_num  # 余数分配给最后一个进程
            if i == 1:
                start_num[i] = 0
            else:
                start_num[i] = end_num[i - 1] + 1
            if i == pool_num - 1:
                end_num[i] = len(filesList)
            else:
                end_num[i] = start_num[i] + list_num - 1
        for i in range(pool_num):
            pool.apply_async(func=process_core, args=(start_num[i], end_num[i]))
            pool.close()  # 关闭进程池，关闭后pool不再接收新的请求任务
            pool.join()  # 等待pool进程池中所有的子进程执行完成，必须放在pool.close()之后
        print('finish all')





