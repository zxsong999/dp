import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def gif():
    file_ori = r"./Data/original"
    file_cur = r"./Data/curve"
    file_vid = r"./Data/video"

    path_list = os.listdir(file_ori)
    images = [imageio.imread(file_ori + '/' + str(img_path)) for img_path in path_list]
    imageio.mimsave("ori.gif", images, "gif", duration=600, loop=1)

    path_list = os.listdir(file_cur)
    images = [imageio.imread(file_cur + '/' + str(img_path)) for img_path in path_list]
    imageio.mimsave("cur.gif", images, "gif", duration=600, loop=1)

    path_list = os.listdir(file_vid)
    images = [imageio.imread(file_vid + '/' + str(img_path)) for img_path in path_list]
    imageio.mimsave("vid.gif", images, "gif", duration=600, loop=1)


# 画图函数
def ed_visualize(all_prop_trans, lane_prop_trans, all_prop_ed, lane_prop_ed, name):
    # 画图轨迹裁剪
    start = all_prop_ed.index[0]
    end = all_prop_ed.index[-1]
    all_prop_trans = all_prop_trans.loc[start+10:end-10]
    all_prop_ed = all_prop_ed.loc[start+10:end-10]

    other_prop_trans = all_prop_trans.loc[all_prop_trans['ID'] != -1]
    other_prop_ed = all_prop_ed.loc[all_prop_ed['ID'] != -1]
    adc_prop_trans = all_prop_trans.loc[all_prop_trans['ID'] == -1]
    adc_prop_ed = all_prop_ed.loc[all_prop_ed['ID'] == -1]

    # 原始数据画图，他车和自车颜色区分
    plt.figure()
    plt.ylim((-10, 10)) # 纵坐标的数值范围
    plt.scatter(other_prop_trans['DX'], other_prop_trans['DY'], c='blue', s=1)
    plt.scatter(adc_prop_trans['DX'], adc_prop_trans['DY'], c='red', s=1)
    for i, row in lane_prop_trans.iterrows():
        # 提取坐标点
        L1, L2, L3 = (row['L1_x'], row['L1_y']), (row['L2_x'], row['L2_y']), (row['L3_x'], row['L3_y'])
        R1, R2, R3 = (row['R1_x'], row['R1_y']), (row['R2_x'], row['R2_y']), (row['R3_x'], row['R3_y'])
        LB, RB = (row['LB_x'], row['LB_y']), (row['RB_x'], row['RB_y'])
        # 绘制图形
        plt.scatter(L1[0], L1[1], c='black', s=1)
        plt.scatter(L2[0], L2[1], c='black', s=1)
        plt.scatter(L3[0], L3[1], c='black', s=1)
        plt.scatter(R1[0], R1[1], c='black', s=1)
        plt.scatter(R2[0], R2[1], c='black', s=1)
        plt.scatter(R3[0], R3[1], c='black', s=1)
        # plt.scatter(LB[0], LB[1], c='black', s=1)
        # plt.scatter(RB[0], RB[1], c='black', s=1)
    plt.savefig(r'Data\pic_display\DPP_before\orgin' + '_' + name + ".jpg")
    plt.close()

    # 拟合后结果
    plt.figure()
    plt.ylim((-10, 10)) # 纵坐标的数值范围
    plt.scatter(other_prop_ed['DX'], other_prop_ed['DY'], c='blue', s=1)
    plt.scatter(adc_prop_ed['DX'], adc_prop_ed['DY'], c='red', s=1)
    c = np.zeros([1, 8])
    for i in range(6):
        st, ed = lane_prop_ed.iloc[i, 0], lane_prop_ed.iloc[i, 1]
        lane_x = np.arange(st, ed, 0.5)
        lane_y = lane_x
        for j in range(lane_prop_ed.shape[1]-2):
            c[0, j] = lane_prop_ed.values[i, 2+j]
        for index, x in enumerate(lane_x):
            lane_y[index] = c[0,0] + c[0,1]*x + c[0,2]*(x**2) + c[0,3]*(x**3) + c[0,4]*(x**4) + c[0,5]*(x**5) + c[0,6]*(x**6) + c[0,7]*(x**7)
        lane_x = np.arange(st, ed, 0.5)
        plt.scatter(lane_x, lane_y, c='black', s=1)
        plt.scatter(lane_x, lane_y, c='black', s=1)
    plt.savefig(r'Data\pic_display\DPP_after\fit' + '_' + name + ".jpg")
    plt.close()
