RFM_Lane_list_10 = ['RFMresult.RFM_TSR.RoadType', 'RFMresult.RFM_TSR.SPL_LaneChanged',
                    'RFMresult.RFM_LINE_CL.bIsLineExist',
                    'name: RFMresult.RFM_ROAD.nLaneNum', 'RFMresult.RFM_LINE_CL.dC3V', 'RFMresult.RFM_ROAD.b_bridge',
                    'RFMresult.RFM_ROAD.b_toll', 'RFMresult.RFM_ROAD.b_traffic_accident',
                    'RFMresult.RFM_ROAD.b_traffic_jam',
                    'RFMresult.RFM_ROAD.b_tunnel', 'RFMresult.RFM_BRANCH_LANE.b_is_available',
                    'RFMresult.RFM_INTERSEC_ALERT.b_is_available',
                    'RFMresult.RFM_INTERSEC_ALERT.f_distance', 'RFMresult.RFM_RIGHT_LANE.en_lane_type',
                    'RFMresult.RFM_LEFT_LANE.en_lane_type',
                    'RFMresult.RFM_CURRENT_LANE.en_lane_type', 'RFMresult.RFM_BRANCH_LANE.en_lane_type']  # dd
HWA_object_10 = ['TOS_LC._0_.Object_ID', 'TOS_LC._1_.Object_ID', 'TOS_LC._2_.Object_ID', 'TOS_LC._3_.Object_ID',
                 'TOS_LC._4_.Object_ID', 'TOS_LC._5_.Object_ID', 'TOS_LC._6_.Object_ID', 'TOS_LC._7_.Object_ID',
                 'TOS_LC._8_.Object_ID', 'TOS_LC._9_.Object_ID', 'TOS_LC._10_.Object_ID']
HWA_object_1 = ['TOS_HWA_RightLane_ObjIndex']
OFM_self_10 = ['autosar_swc_objfusion_rtB.SubOFM.IM_vxvRefMs', 'autosar_swc_objfusion_rtB.SubOFM.IM_axRefMs2',
               'autosar_swc_objfusion_rtB.SubOFM.IM_ayvRefMs2',
               'autosar_swc_objfusion_rtB.SubOFM.IM_kapTraj']
OFM_EGO_10 = ['ego_pack.vxvRefMs', 'ego_pack.axRefMs2', 'ego_pack.ayvRefMs2', 'ego_pack.SteerAngleOffset_deg',
              'ego_pack.kapTraj']



def OFM_object_10(i):
    OFM_object_10 = ['autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_ID._{}_'.format(i),'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_Type._{}_'.format(i),
                    'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_DX._{}_'.format(i),'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_DY._{}_'.format(i),
                     'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_VX._{}_'.format(i),'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_VY._{}_'.format(i),
                     'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_AX._{}_'.format(i),'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_AY._{}_'.format(i),
                     'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_Length._{}_'.format(i),'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_Width._{}_'.format(i),
                     'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_YawAngle._{}_'.format(i),'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_MovingState._{}_'.format(i),
                     'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_Orientation._{}_'.format(i),'autosar_swc_objfusion_rtB.SubOFM.OC_OFM_OBJS.Object_Age._{}_'.format(i)]
    return OFM_object_10

def EMP_object_10(i):
    EMP_object_10 = ['EMP_Object_ID._{}_'.format(i),'EMP_Object_Fus_Type._{}_'.format(i), 'EMP_Object_Type._{}_'.format(i),
                     'EMP_Object_Length._{}_'.format(i),'EMP_Object_Width._{}_'.format(i),
                     'EMP_Object_Hight._{}_'.format(i),
                     'EMP_Object_DX._{}_'.format(i),'EMP_Object_DY._{}_'.format(i),
                    'EMP_Object_VX._{}_'.format(i),'EMP_Object_VY._{}_'.format(i),'EMP_Object_AX._{}_'.format(i),'EMP_Object_AY._{}_'.format(i),
                     'EMP_Object_ExistProb._{}_'.format(i), 'EMP_OFM_ObjValid_Flag._{}_'.format(i),
                     'EMP_Object_Age._{}_'.format(i),'EMP_Object_MovingState._{}_'.format(i),'EMP_Object_NextMrrId._{}_'.format(i),
                     'EMP_Object_ObstacleProb._{}_'.format(i)
                     ]
    return EMP_object_10


def OFM_obj_160(i):
    OFM_obj_160 = ['EMP_OFM_OBJS.Object_ID._{}_'.format(i), 'EMP_OFM_OBJS.Object_Fus_Type._{}_'.format(i), 'EMP_OFM_OBJS.Object_Type._{}_'.format(i),
                   'EMP_OFM_OBJS.Object_Length._{}_'.format(i),  'EMP_OFM_OBJS.Object_Width._{}_'.format(i),  'EMP_OFM_OBJS.Object_Hight._{}_'.format(i),
                   'EMP_OFM_OBJS.Object_DX._{}_'.format(i), 'EMP_OFM_OBJS.Object_DY._{}_'.format(i),  'EMP_OFM_OBJS.Object_AX._{}_'.format(i),
                   'EMP_OFM_OBJS.Object_AY._{}_'.format(i), 'EMP_OFM_OBJS.Object_VX._{}_'.format(i),  'EMP_OFM_OBJS.Object_VY._{}_'.format(i),
                   'EMP_OFM_OBJS.Object_ExistProb._{}_'.format(i), 'EMP_OFM_OBJS.Object_Age._{}_'.format(i), 'EMP_OFM_OBJS.Object_MovingState._{}_'.format(i),
                   'EMP_OFM_OBJS.Object_NextMrrId._{}_'.format(i), 'EMP_OFM_OBJS.Object_ObstacleProb._{}_'.format(i), 'EMP_OFM_OBJS.Object_YawAngle._{}_'.format(i),
                   'EMP_OFM_OBJS.Object_HeadingAngle._{}_'.format(i)
                   ]
    return OFM_obj_160


def EMP_self_property():
    EMP_self_property = ['EMP_vxvRefMs', 'EMP_axRefMs2', 'EMP_ayvRefMs2', 'EMP_psiDtOpt', 'EMP_kapTraj']
    #                     自车参考车速    自车纵向加速度    自车横向加速度   自车横摆角速度     道路曲率
    return EMP_self_property


def lane_RFM_property():
    lane_RFM_property = ['SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dC0V',
                         'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dC1V',
                         'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dC2V',
                         'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dC3V',
                         'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dDisStart',
                         'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dDisFront',
                         'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dQuality',
                         'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.bIsLineExist',
                         'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dC0V',
                         'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dC1V',
                         'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dC2V',
                         'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dC3V',
                         'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dDisStart',
                         'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dDisFront',
                         'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dQuality',
                         'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.bIsLineExist',
                         'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dC0V',
                         'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dC1V',
                         'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dC2V',
                         'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dC3V',
                         'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dDisStart',
                         'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dDisFront',
                         'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dQuality',
                         'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.bIsLineExist',
                         'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dC0V',
                         'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dC1V',
                         'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dC2V',
                         'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dC3V',
                         'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dDisStart',
                         'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dDisFront',
                         'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dQuality',
                         'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.bIsLineExist',
                         'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dC0V',
                         'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dC1V',
                         'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dC2V',
                         'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dC3V',
                         'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dDisStart',
                         'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dDisFront',
                         'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dQuality',
                         'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.bIsLineExist',
                         'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dC0V',
                         'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dC1V',
                         'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dC2V',
                         'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dC3V',
                         'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dDisStart',
                         'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dDisFront',
                         'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dQuality',
                         'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.bIsLineExist',
                         'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dC0V',
                         'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dC1V',
                         'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dC2V',
                         'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dC3V',
                         'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dDisStart',
                         'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dDisFront',
                         'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dQuality',
                         'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.bIsLineExist',
                         'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dC0V',
                         'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dC1V',
                         'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dC2V',
                         'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dC3V',
                         'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dDisStart',
                         'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dDisFront',
                         'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dQuality',
                         'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.bIsLineExist',
                         ]
    return lane_RFM_property

# 列名


cols_lane_prop = ['timestamps', 'L1_C0', 'L1_C1', 'L1_C2', 'L1_C3', 'L1_start', 'L1_end', 'L1_quality', 'L1_exist',
                         'L2_C0', 'L2_C1', 'L2_C2', 'L2_C3', 'L2_start', 'L2_end','L2_quality', 'L2_exist',
                         'L3_C0', 'L3_C1', 'L3_C2', 'L3_C3', 'L3_start', 'L3_end','L3_quality', 'L3_exist',
                         'R1_C0', 'R1_C1', 'R1_C2', 'R1_C3', 'R1_start', 'R1_end','R1_quality', 'R1_exist',
                         'R2_C0', 'R2_C1', 'R2_C2', 'R2_C3', 'R2_start', 'R2_end','R2_quality', 'R2_exist',
                         'R3_C0', 'R3_C1', 'R3_C2', 'R3_C3', 'R3_start', 'R3_end','R3_quality', 'R3_exist',
                         'LB_C0', 'LB_C1', 'LB_C2', 'LB_C3', 'LB_start', 'LB_end','LB_quality', 'LB_exist',
                         'RB_C0', 'RB_C1', 'RB_C2', 'RB_C3', 'RB_start', 'RB_end','RB_quality', 'RB_exist'
                         ]


def lane_rename(lane_prop_list):
    lane_prop_list = lane_prop_list.rename(columns={'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dC0V': 'L1_C0',
                                                    'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dC1V': 'L1_C1',
                                                    'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dC2V': 'L1_C2',
                                                    'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dC3V': 'L1_C3',
                                                    'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dDisStart': 'L1_start',
                                                    'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dDisFront': 'L1_end',
                                                    'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.dQuality': 'L1_quality',
                                                    'SIC_InfofusionIf_RFM_LINE_L1_RFM_LINE_BUS_data.bIsLineExist': 'L1_exist',
                                                    'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dC0V':'L2_C0',
                                                    'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dC1V':'L2_C1',
                                                    'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dC2V':'L2_C2',
                                                    'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dC3V':'L2_C3',
                                                    'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dDisStart':'L2_start',
                                                    'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dDisFront':'L2_end',
                                                    'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.dQuality':'L2_quality',
                                                    'SIC_InfofusionIf_RFM_LINE_L2_RFM_LINE_BUS_data.bIsLineExist': 'L2_exist',
                                                    'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dC0V':'L3_C0',
                                                    'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dC1V':'L3_C1',
                                                    'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dC2V':'L3_C2',
                                                    'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dC3V':'L3_C3',
                                                    'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dDisStart':'L3_start',
                                                    'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dDisFront':'L3_end',
                                                    'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.dQuality':'L3_quality',
                                                    'SIC_InfofusionIf_RFM_LINE_L3_RFM_LINE_BUS_data.bIsLineExist': 'L3_exist',
                                                    'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dC0V':'R1_C0',
                                                    'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dC1V':'R1_C1',
                                                    'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dC2V':'R1_C2',
                                                    'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dC3V':'R1_C3',
                                                    'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dDisStart':'R1_start',
                                                    'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dDisFront':'R1_end',
                                                    'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.dQuality':'R1_quality',
                                                    'SIC_InfofusionIf_RFM_LINE_R1_RFM_LINE_BUS_data.bIsLineExist': 'R1_exist',
                                                    'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dC0V':'R2_C0',
                                                    'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dC1V':'R2_C1',
                                                    'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dC2V':'R2_C2',
                                                    'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dC3V':'R2_C3',
                                                    'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dDisStart':'R2_start',
                                                    'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dDisFront':'R2_end',
                                                    'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.dQuality':'R2_quality',
                                                    'SIC_InfofusionIf_RFM_LINE_R2_RFM_LINE_BUS_data.bIsLineExist': 'R2_exist',
                                                    'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dC0V':'R3_C0',
                                                    'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dC1V':'R3_C1',
                                                    'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dC2V':'R3_C2',
                                                    'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dC3V':'R3_C3',
                                                    'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dDisStart':'R3_start',
                                                    'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dDisFront':'R3_end',
                                                    'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.dQuality':'R3_quality',
                                                    'SIC_InfofusionIf_RFM_LINE_R3_RFM_LINE_BUS_data.bIsLineExist': 'R3_exist',
                                                    'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dC0V':'LB_C0',
                                                    'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dC1V':'LB_C1',
                                                    'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dC2V':'LB_C2',
                                                    'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dC3V':'LB_C3',
                                                    'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dDisStart':'LB_start',
                                                    'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dDisFront':'LB_end',
                                                    'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.dQuality':'LB_quality',
                                                    'SIC_InfofusionIf_RFM_LINE_LB_RFM_LINE_BUS_data.bIsLineExist':'LB_exist',
                                                    'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dC0V':'RB_C0',
                                                    'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dC1V':'RB_C1',
                                                    'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dC2V':'RB_C2',
                                                    'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dC3V':'RB_C3',
                                                    'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dDisStart':'RB_start',
                                                    'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dDisFront':'RB_end',
                                                    'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.dQuality':'RB_quality',
                                                    'SIC_InfofusionIf_RFM_LINE_RB_RFM_LINE_BUS_data.bIsLineExist':'RB_exist'})
    return lane_prop_list


def obj_rename(obj_160_prop, i):
    obj_160_prop = obj_160_prop.rename(columns={'EMP_OFM_OBJS.Object_ID._{}_'.format(i): 'ID',
                                                'EMP_OFM_OBJS.Object_Fus_Type._{}_'.format(i): 'Fus_Type',
                                                'EMP_OFM_OBJS.Object_Type._{}_'.format(i): 'Type',
                                                'EMP_OFM_OBJS.Object_Length._{}_'.format(i): 'Length',
                                                'EMP_OFM_OBJS.Object_Width._{}_'.format(i): 'Width',
                                                'EMP_OFM_OBJS.Object_Hight._{}_'.format(i): 'Hight',
                                                'EMP_OFM_OBJS.Object_DX._{}_'.format(i): 'DX',
                                                'EMP_OFM_OBJS.Object_DY._{}_'.format(i): 'DY',
                                                'EMP_OFM_OBJS.Object_AX._{}_'.format(i): 'AX',
                                                'EMP_OFM_OBJS.Object_AY._{}_'.format(i): 'AY',
                                                'EMP_OFM_OBJS.Object_VX._{}_'.format(i): 'VX',
                                                'EMP_OFM_OBJS.Object_VY._{}_'.format(i): 'VY',
                                                'EMP_OFM_OBJS.Object_ExistProb._{}_'.format(i): 'ExistProb',
                                                'EMP_OFM_OBJS.Object_Age._{}_'.format(i): 'Age',
                                                'EMP_OFM_OBJS.Object_MovingState._{}_'.format(i): 'MovingState',
                                                'EMP_OFM_OBJS.Object_NextMrrId._{}_'.format(i): 'NextMrrId',
                                                'EMP_OFM_OBJS.Object_ObstacleProb._{}_'.format(i): 'ObstacleProb',
                                                'EMP_OFM_OBJS.Object_YawAngle._{}_'.format(i): 'YawAngle',
                                                'EMP_OFM_OBJS.Object_HeadingAngle._{}_'.format(i): 'HeadingAngle'})
    return obj_160_prop


def cols_annofile():
    cols_anno = ['timestamps','R_type', 'obj_self']
    for i in range(1,161):
        x = 'obj_' + str(i)
        cols_anno.append(x)
    return cols_anno


def cols_traj():
    cols_traj = ['timestamps', 'ID', 'Type', 'hight', 'width', 'length']
    for i in range(50):
        x = 'x' + str(i)
        cols_traj.append(x)
    for i in range(50):
        y = 'y' + str(i)
        cols_traj.append(y)
    for i in range(50):
        mask_traj = 'mask_traj' + str(i)
        cols_traj.append(mask_traj)
    return cols_traj


cols_obj_160 = ['timestamps', 'ID', 'Type', 'Fus_Type', 'Length', 'Width', 'Hight', 'Age', 'MovingState',
                'NextMrrId', 'ObstacleProb', 'ExistProb', 'DX', 'DY', 'VX', 'VY', 'AX', 'AY', 'HeadingAngle',
                'YawAngle']


cols_Refms = ['timestamps', 'ID', 'EMP_vxvRefMs', 'EMP_axRefMs2', 'EMP_ayvRefMs2', 'EMP_psiDtOpt', 'EMP_kapTraj']