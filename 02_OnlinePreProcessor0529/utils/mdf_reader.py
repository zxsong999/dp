from asammdf import MDF
import numpy as np
import pandas as pd
import re
from asammdf.blocks.utils import extract_xml_comment, MdfException
# from debug.log import log

class Mf4Reader():
    def __init__(self, mf4_path):
        self.mf4_path = mf4_path
        self.reader = MDF(mf4_path)
        self.cdb = self.reader.channels_db
        self.channelValsCacheMap = dict()
        self.group_index_map = dict()
        self.groupDataframeCacheMap = dict()

    def filter(self, channel_namelist):
        """
        filter out the channels to be uploaded
        Parameters
        channel_namelist [list] list of channels' name to keep
        """
        self.reader = self.reader.filter(channel_namelist)
        self.cdb = self.reader.channels_db

    def get_comment_text(self):
        """
        get text of MDF'comment
        """
        return extract_xml_comment(self.reader.header.comment)

    def get_start_end_time(self, max_interval=False):
        """
        get all channels start_time and end_time and return a time interval
        Parameters
        max_interval [bool] return a maxium time interval
        """
        t_min = []
        t_max = []

        for idx in self.reader.virtual_groups:
            master = self.reader.get_master(idx, record_offset=0)
            if master.size:
                t_min.append(master[0])
                t_max.append(master[-1])

        t_min_short = np.amin(t_min)
        t_min_long = np.amax(t_min)
        t_max_short = np.amin(t_max)
        t_max_long = np.amax(t_max)
        if max_interval:
            # log.info('start:%f end:%f' % (t_min_short, t_max_long))
            return (t_min_short, t_max_long)
        # log.info('start:%f end:%f' % (t_min_long, t_max_short))
        return (t_min_long, t_max_short)

    def cut(self, start=None, stop=None, whence=0, include_ends=True, time_from_zero=False):
        """
        Parameters
        start [float] start time, default None. If None then the start of measurement is used
        stop [float] stop time, default None. If None then the end of measurement is used
        whence [int] how to search for the start and stop values
        ? 0 : absolute
        ? 1 : relative to first timestamp
        include_ends [bool] include the start and stop timestamps after cutting the signal. If start
        and stop are found in the original timestamps, then the new samples will be computed
        using interpolation. Default True
        time_from_zero [bool] start time stamps from 0s in the cut measurement
        """
        self.reader = self.reader.cut(
            start=start, stop=stop, whence=0, include_ends=include_ends, time_from_zero=time_from_zero)

    def configInterpMode(self, interpMode=None):
        """
        Parameters
        interpMode [int] interpolation mode for integer channels:
        ? 0 - repeat previous sample
        ? 1 - use linear interpolation
        """
        self.reader.configure(integer_interpolation=interpMode)

    def resample(self, raster, time_from_zero=False):
        """
        resample all channels using the given raster. See configure to select the interpolation method for interger
        channels
        Parameters
        raster [float | np.array | str] new raster that can be
        ? a float step value
        ? a channel name who��s timestamps will be used as raster
        ? an array
        time_from_zero [bool] start time stamps from 0s in the cut measurement
        """
        self.reader = self.reader.resample(
            raster=raster, time_from_zero=time_from_zero)

    def timesync(self, raster=0.02):
        (start, end) = self.get_start_end_time()
        self.cut(start=start, stop=end, whence=0,
                 include_ends=True, time_from_zero=False)
        self.configInterpMode(interpMode=0)
        self.resample(raster=raster)

    def channel_to_group_map(self):
        for gitem in self.reader.groups:
            print(gitem.id, ":{ ")
            for channel in gitem["channels"]:
                print(channel.name, ", ")
            print("}\n")

    def contain_channels(self, channel_list):
        for channel in channel_list:
            if not self.reader.__contains__(channel):
                channel_list.remove(channel)
        return channel_list

    def judge_channel_validity(self,channel,group_idx):
        channel_info = list()
        channel_info.append(channel)
        channel_info.append(group_idx)
        tchannels = list()
        tchannels.append(channel_info)
        df = self.reader.to_dataframe(channels=tchannels, time_from_zero=False, empty_channels='skip', raw=False, ignore_value2text_conversions=True)
        if len(df.to_dict()) == 0:
            return False,channel_info
        return True,channel_info
    def channels_filter(self,channels):
        cur_channel = list()
        for channel in channels:
            idx_lists = self.reader.whereis(channel)

            if len(idx_lists) == 1:
                cur_channel.append(channel)
            elif len(idx_lists)>1:
                channel_info = list()
                flag = False
                for i in range(len(idx_lists)):
                    (flag,channel_info) = self.judge_channel_validity(channel,idx_lists[i][0])
                    if flag:
                        cur_channel.append(channel_info)
                        break
            # else:
            #     log.warning("Can not find the key: {}".format(', '.join(channel)))
        return cur_channel
    # TODO: add arguments with deal channel name repeat
    def get_dataFrame2(self, channels):

        cur_channels = self.channels_filter(channels)
        if len(cur_channels) > 0:
            return self.reader.to_dataframe(channels=cur_channels, time_from_zero=False, empty_channels='skip', raw=False, ignore_value2text_conversions=True)
        # log.warning("Can not find the key: {}".format(', '.join(channels)))
        return pd.DataFrame()


    # TODO: add arguments
    def get_dataFrame(self, channels):
        cur_channels = [channel for channel in channels if channel in self.reader.channels_db]
        if len(cur_channels) > 0:
            return self.reader.to_dataframe(channels=cur_channels, time_from_zero=False, empty_channels='skip', raw=False, ignore_value2text_conversions=True)
        # log.warning("Can not find the key: {}".format(', '.join(channels)))
        return pd.DataFrame()

    def get_group_index_map(self):
        for idx, gitem in enumerate(self.reader.groups):
            gname = gitem['channel_group'].acq_name
            self.group_index_map[gname] = idx

    def get_group_source_info(self, group_idx):
        _dataGroup = self.reader.groups[group_idx]
        _channelGroup = _dataGroup["channel_group"]
        return extract_xml_comment(_channelGroup.acq_source.comment)

    def get_info_in_details(self):
        print("mdf name: \n",  self.reader.name)
        print("mdf version: \n",  self.reader.version)
        print(self.reader.info())

    def get_channel_name_list(self):
        namelist = list()
        for ci in self.cdb:
            namelist.append(ci)
        return namelist

    def select_read(self, group_name, channel_namelist):
        selectedSignals = list()
        channelValuesMap = dict()
        selectedSignals = self.reader.select(
            channel_namelist, copy_master=False, ignore_value2text_conversions=True)
        for idx, sig in enumerate(selectedSignals):
            channel_name = sig.name
            channelValuesMap[channel_name] = selectedSignals[idx]
        print("debug**** size of channelValuesMap",
              len(channelValuesMap.keys()), ", with group_name: ", group_name)
        self.channelValsCacheMap[group_name] = channelValuesMap

    def init_multi_read(self, group_namelist):
        for group_name in group_namelist:
            try:
                group_index = self.group_index_map[group_name]
                self.groupDataframeCacheMap[group_name] = self.reader.get_group(
                    group_index, ignore_value2text_conversions=True)
            except Exception:
                print("can't get group", group_name)
                exit

    def updateCacheData(self, group_name, time, index):
        cacheMap = dict()
        for channel_name, channel_raw_data in self.channelValsCacheMap[group_name].items():
            # TODO: there is chance idx >= len(ts), which lead to samples[idx] out of boundary, here temply set as 0
            #  number_ts:  1153  keys: MRR_ObjObstacleProb_1
            #  number_ts:  1152  keys: MRR_ObjObstacleProb_3
            if index >= len(channel_raw_data.timestamps):
                cacheMap[channel_name] = 0
                continue
            vals = channel_raw_data.samples[index]
            cacheMap[channel_name] = vals
        return cacheMap

    def seek_closest_timestamp_index(self, ts, time, flag="SeekLessOrEqual"):
        dt = ts[1] - ts[0]
        idx = int((time - ts[0])/dt)
        return idx

    def get_channel_all_signals_by_name(self, channel_name):
        """
        return a Signal type https://asammdf.readthedocs.io/en/master/api.html#signal
        """
        sig = None
        if channel_name in self.cdb:
            channel_item = self.cdb[channel_name]
            if '_ObjID_' in channel_name or '_ObjType_' in channel_name or '_ObjMotionType_' in channel_name or '_ObjMotionDirection_' in channel_name:
                sig = self.reader.get(
                    channel_name, channel_item[0][0], channel_item[0][1], raw=True)  # uint8(raw vals)
            else:
                sig = self.reader.get(
                    channel_name, channel_item[0][0], channel_item[0][1])
        if sig:
            return sig
        else:
            return None

    # TODO: opt with only one reader.get()
    def get_channel_data_by_name(self, channel_name, time=0.13):
        sig_content = None
        if channel_name in self.cdb:
            channel_item = self.cdb[channel_name]
            sig_content = self.reader.get(
                channel_name, channel_item[0][0], channel_item[0][1], raw=True)
            found_idx = self.seek_closest_timestamp_index(
                sig_content.timestamps, time)  # type(found_idx) int
            try:
                if found_idx >= len(sig_content.samples):
                    return sig_content.samples[-1]
                return sig_content.samples[found_idx]
            except Exception:
                print("timestamp not found\n")
        return sig_content

    def get_relative_time(self):
        st = self.reader.start_time
        print("canape time:", st.strftime('%Y-%m-%d %H:%M:%S %f'))
        pat = r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}"
        res = re.search(pat, self.mf4_path)
        save_date = list()
        try:
            save_time = res.group()
            print("save time", save_time)
            save_date = re.findall("\d+", save_time)
        except Exception:
            print("mf4 path error")
        dhr = int(save_date[3]) - int(st.hour)
        dmin = int(save_date[4]) - int(st.minute)
        dsec = int(save_date[5]) - int(st.second)

        return dhr*3600 + dmin*60 + dsec

    def get_group_sampling_rates(self):
        sampling_rates = dict()
        for i, group in enumerate(self.reader.groups):
            master = self.reader.get_master(i)
            gname = group['channel_group'].acq_name
            if len(master) > 1:
                rate = np.mean(np.diff(master))
            else:
                rate = 0
            sampling_rates[gname] = rate
        return sampling_rates

    def get_f_video(self):
        channel_name = "F_Video"
        video_time = self.reader.get(channel_name).samples
        singal_time = self.reader.get(channel_name).timestamps
        df = {"timestamps": singal_time, "value": video_time}
        df = pd.core.frame.DataFrame(df)
        df_result = df.set_index("timestamps")
        return df_result
