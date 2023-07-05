import math
import numpy as np


def boundingbox(dx, dy, width, length, heading_angle):
    if heading_angle == 'None':
        heading_angle = 0
        LU_x = (dx - 0.5*width*math.sin(heading_angle) + length*math.cos(heading_angle))*1.1
        LU_y = (dy + 0.5*width*math.cos(heading_angle) + length*math.sin(heading_angle))*1.1
        LL_x = (dx - 0.5+width*math.sin(heading_angle))*1.1
        LL_y = (dy + 0.5*width*math.cos(heading_angle))*1.1
        RU_x = (dx + 0.5*width*math.sin(heading_angle) + length*math.cos(heading_angle))*1.1
        RU_y = (dy - 0.5*width*math.cos(heading_angle) + length*math.sin(heading_angle))*1.1
        RL_x = (dx + 0.5+width*math.sin(heading_angle))*1.1
        RL_y = (dy - 0.5*width*math.cos(heading_angle))*1.1
    else:
        LU_x = dx - 0.5*width*math.sin(heading_angle) + length*math.cos(heading_angle)
        LU_y = dy + 0.5*width*math.cos(heading_angle) + length*math.sin(heading_angle)
        LL_x = dx - 0.5+width*math.sin(heading_angle)
        LL_y = dy + 0.5*width*math.cos(heading_angle)
        RU_x = dx + 0.5*width*math.sin(heading_angle) + length*math.cos(heading_angle)
        RU_y = dy - 0.5*width*math.cos(heading_angle) + length*math.sin(heading_angle)
        RL_x = dx + 0.5+width*math.sin(heading_angle)
        RL_y = dy - 0.5*width*math.cos(heading_angle)

    return LU_x, LU_y, LL_x, LL_y, RU_x, RU_y, RL_x, RL_y


def intersection_lane(XX_y, XX_x, c0, c1, c2, c3):
    dx = XX_x
    lane_dy = c3*dx*dx*dx + c2*dx*dx + c1*dx + c0
    if XX_y > lane_dy:
        flag = 1
    else:
        flag = 2
    return flag


def sinc(DeltaAngle):  # 计算sin函数

    if abs(DeltaAngle) < 0.1:
        Taylor_poly = 1 - ((DeltaAngle * DeltaAngle) / 6) * (1 - ((DeltaAngle * DeltaAngle) / 20) * (
                    1 - ((DeltaAngle * DeltaAngle) / 42) * (1 - ((DeltaAngle * DeltaAngle) / 72))))
    else:
        Taylor_poly = (math.sin(DeltaAngle)) / DeltaAngle
    return Taylor_poly


def intersection_lane(XX_y, XX_x, c0, c1, c2, c3):
    dx = XX_x
    lane_dy = c3 * dx * dx * dx + c2 * dx * dx + c1 * dx + c0
    if XX_y > lane_dy:
        flag = 1
    else:
        flag = 2
    return flag


# 定义车道线方程
def lane_line(x, c0, c1, c2, c3):
    return c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3


# 计算车辆和车道线的交点
def intersection(lane, bbox):
    xvals = [bbox[0], bbox[2], bbox[4], bbox[6]]
    yvals = [bbox[1], bbox[3], bbox[5], bbox[7]]
    # 如果四个顶点的y值都小于车道线的最小值或都大于最大值，则认为车辆没有与车道线相交
    if (np.all(yvals < lane_line(np.min(xvals), *lane)) or np.all(yvals > lane_line(np.max(xvals), *lane))):
        return 0
    # 计算交点
    x_intersect = np.roots([lane[3], lane[2], lane[1], lane[0] - yvals[0]])
    y_intersect = lane_line(x_intersect, *lane)
    # 取第一个交点作为最终结果

    y_center = (bbox[1] + bbox[3] + bbox[5] + bbox[7]) / 4

    if y_intersect[0] < y_center:
        # 交点在车辆左侧，认为车辆左变道
        return 1
    elif y_intersect[0] > y_center:
        # 交点在车辆右侧，认为车辆右变道
        return 2
    else:
        # 车辆压线，可能是变道未完成
        return 3


def poly_func(x, *coefficients):
    # 使用poly1d函数创建多项式函数对象
    poly_function = np.poly1d(coefficients)
    return poly_function(x)




