class DefaultConfig(object):
    # 主程序参数
    file_path = r'Data/data_mf4'
    save_path = r'Data/data_pkl'
    pkl_path = r'Data/data_pkl'
    tar_type = 'mf4'  # 要处理的文件类型
    parallel_compute = 'off'    # 是否并行计算 on为是 off为否
    pool_num = 2  # 并行进程

    # 滑动窗口参数
    window_size = 40  # 时间窗口为16秒（320帧）
    distance_window = 1  # 自车行驶距离为50米
    forward = 20  # 下一个窗口向前推进的距离


    # 转模型输入参数
    velocity_threshold = 1  # 速度阈值 Number of timesteps the track should exist to be considered in social context
    exist_threshold = 20    # 存在阈值 # index of the sorted velocity to look at, to call it as stationary
    obs_len = 20            # 观测长度

    # 目标筛选参数
    max_dx = 100  # 前方最大距离
    max_dy = 10  # 左右最大距离

    # 车道线有无判定
    l_q = 0.35
    l_len = 150




    d_slot = 100  # data slot：100帧对应5s的数据；
    d_his = 40  # 数据段分为历史（data_history）和未来(data_future)
    d_fut = 60  # 2s预测3s  d_fut = d_slot-d_his
    slide_t = 100  # slide time 滑动窗口时间 40帧对应2s；