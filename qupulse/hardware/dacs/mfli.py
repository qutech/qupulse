""" This file contains a qupulse driver for the DAQ module of an Zuerich Instruments MFLI.

May lines of code have been adapted from the zihdawg driver.

"""
import dataclasses
import logging
import time
import traceback
import warnings
from typing import Dict, Tuple, Iterable, Union, List, Set, Any, Sequence, Mapping, Optional, Literal

import numpy as np
import xarray as xr

from qupulse.hardware.dacs.dac_base import DAC
from qupulse.utils.types import TimeType

try:
    # zhinst fires a DeprecationWarning from its own code in some versions...
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        import zhinst.utils
except ImportError:
    warnings.warn('Zurich Instruments LabOne python API is distributed via the Python Package Index. Install with pip.')
    raise

try:
    from zhinst import core as zhinst_core
except ImportError:
    # backward compability
    from zhinst import ziPython as zhinst_core

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class TriggerSettings:
    trigger_input: str
    trigger_count: int
    edge: Literal['rising', 'falling', 'both']
    level: float
    delay: float
    post_delay: float
    measurement_count: Union[int, float]

    def is_endless(self):
        return np.isinf(self.measurement_count) or self.measurement_count == 0



def postprocessing_crop_windows(
                serial:str,
                recorded_data: Mapping[str, List[xr.DataArray]],
                program: "MFLIProgram",
                fail_on_empty: bool = True, average_window:bool=False) -> Mapping[str, Mapping[str, List[Union[float, xr.DataArray]]]]:
    """ This function parses the recorded data and extracts the measurement masks
    """

    # the first dimension of channel_data is expected to be the history of multiple not read data points. This will
    # be handled as multiple entries in a list. This will then not make too much sense, if not every channel as this
    # many entries. If this is the case, they will be stacked, such that for the last elements it fits.
    # TODO do this based on the timestamps and not the indices. That might be more sound than just assuming that.

    # applying measurement windows and optional operations
    # TODO implement operations

    # targeted structure:
    # results[<mask_name>][<channel>] -> [data]

    masked_data = {}

    # the MFLI returns a list of measurements. We only proceed with the last ones from this list. One might want to
    # iterate over that and process all of them.This feature might be useful if after some measurements no read()
    # operation is called. Then with the later read, the data is returned.
    # TODO this might be more elegantly implemented or handled using yields!
    shot_index = 0  # TODO make this more flexible to not lose things

    for window_name, (begins, lengths) in program.windows.items():
        data_by_channel = {}
        # _wind = program["windows"][window_name]
        for ci, _cn in enumerate(program.channel_mapping[window_name]):
            cn = f"/{serial}/{_cn}".lower()

            if len(recorded_data[cn]) <= shot_index:
                # then we do not have data for this shot_index, which is intended to cover multiple not yet collected measurements. And thus will not have anything to save.
                warnings.warn(
                    f"for channel '{cn}' only {len(recorded_data[cn])} shots are given. This does not allow for taking element [-1-{shot_index}]")
                continue
            applicable_data = recorded_data[cn][-1 - shot_index]
            applicable_data = applicable_data.where(~np.isnan(applicable_data), drop=True)

            if len(applicable_data) == 0 or np.product([*applicable_data.shape]) == 0:
                if fail_on_empty:
                    raise ValueError(f"The received data for channel {_cn} is empty.")
                else:
                    warnings.warn(f"The received data for channel {_cn} is empty.")
                    continue

            extracted_data = []
            for b, l in zip(begins, lengths):
                # _time_of_first_not_nan_value = applicable_data.where(~np.isnan(applicable_data), drop=True)["time"][:, 0].values

                _time_of_first_not_nan_value = applicable_data["time"][:, 0].values

                time_of_trigger = -1 * applicable_data.attrs["gridcoloffset"][
                    0] * 1e9 + _time_of_first_not_nan_value

                foo = applicable_data.where((applicable_data["time"] >= (time_of_trigger + b)[:, None]) & (
                        applicable_data["time"] <= (time_of_trigger + b + l)[:, None]), drop=False)
                if not average_window:
                    foo = foo.copy()
                    foo2 = foo.where(~np.isnan(foo), drop=True)
                    rows_with_data = np.sum(~np.isnan(foo), axis=-1) > 0
                    foo2["time"] -= time_of_trigger[rows_with_data, None]
                    extracted_data.append(foo2)
                else:
                    extracted_data.append(np.nanmean(foo))

            data_by_channel.update({cn: extracted_data})
        masked_data[window_name] = data_by_channel

    return masked_data


def postprocessing_average_within_windows(
                serial:str,
                recorded_data: Mapping[str, List[xr.DataArray]],
                program: "MFLIProgram",
                fail_on_empty: bool = True) -> Mapping[str, Mapping[str, List[float]]]:
    """ This function returns one float per window that averages each channel individually for that window.
    """

    return postprocessing_crop_windows(
                serial = serial,
                recorded_data = recorded_data,
                program = program,
                fail_on_empty = fail_on_empty, 
                average_window = True)


@dataclasses.dataclass
class MFLIProgram:
    default_channels: Optional[Set[str]] = dataclasses.field(default=None)
    channel_mapping: Optional[Dict[str, Set[str]]] = dataclasses.field(default=None)
    windows: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = dataclasses.field(default=None)
    trigger_settings: Optional[TriggerSettings] = dataclasses.field(default=None)
    other_settings: Dict[str, Any] = dataclasses.field(default_factory=dict)
    operations: Any = dataclasses.field(default=postprocessing_crop_windows)

    def get_minimal_duration(self) -> float:
        return max(np.max(begins + lengths) for (begins, lengths) in self.windows.values())

    def required_channels(self) -> Set[str]:
        channels = set()
        for window_name in self.windows:
            channels |= self.channel_mapping.get(window_name, self.default_channels)
        return channels

    def merge(self, other: 'MFLIProgram') -> 'MFLIProgram':

        new_program = MFLIProgram()

        new_program.default_channels = self.default_channels
        if new_program.default_channels is None:
            new_program.default_channels = other.default_channels
        elif isinstance(new_program.default_channels, set):
            new_program.default_channels = new_program.default_channels.union(other.default_channels)

        new_program.channel_mapping = {**self.channel_mapping}
        if other.channel_mapping is not None:
            for k, v in other.channel_mapping.items():
                new_program.channel_mapping[k].update(v)

        if self.windows is not None or other.windows is not None:
            new_program.windows = {}

            def add_to_windows(name, begins, lengths):
                if name not in new_program.windows:
                    new_program.windows[name] = [[], []]
                for b, l in zip(begins, lengths):
                    if b not in new_program.windows[name][0]:
                        new_program.windows[name][0].append(b)
                        new_program.windows[name][1].append(l)

            if self.windows is not None:
                for wn, v in self.windows.items():
                    add_to_windows(wn, v[0], v[1])
            if other.windows is not None:
                for wn, v in other.windows.items():
                    add_to_windows(wn, v[0], v[1])
            
            for k, v in new_program.windows.items():
                new_program.windows[k][0] = np.array(new_program.windows[k][0])
                new_program.windows[k][1] = np.array(new_program.windows[k][1])

        if self.trigger_settings is not None:
            new_program.trigger_settings = self.trigger_settings
        if other.trigger_settings is not None:
            new_program.trigger_settings = other.trigger_settings

        new_program.other_settings.update(self.other_settings)
        new_program.other_settings.update(other.other_settings)

        if self.operations is not None:
            new_program.operations = self.operations
        if other.operations is not None:
            new_program.operations = other.operations

        return new_program


class MFLIDAQ(DAC):
    """ This class contains the driver for using the DAQ module of an Zuerich Instruments MFLI with qupulse.
    """

    def __init__(self,
                 api_session: zhinst_core.ziDAQServer,
                 device_props: Dict,
                 reset: bool = False,
                 timeout: float = 20) -> None:
        """
        :param reset:             Reset device before initialization
        :param timeout:           Timeout in seconds for uploading
        """
        self.api_session = api_session
        self.device_props = device_props
        self.default_timeout = timeout
        self.serial = device_props["deviceid"]

        self.daq = None
        self._init_daq_module()

        self.force_update_on_arm: bool = True
        self.assumed_minimal_sample_rate: Union[float, None] = None  # in units of Sa/s

        self.daq_read_return = {}
        self.read_memory: Dict[str, List[xr.DataArray]] = {}  # self.read_memory[<path/channel>]:List[xr.DataArray]

        if reset:
            # Create a base configuration: Disable all available outputs, awgs, demods, scopes,...
            zhinst.utils.disable_everything(self.api_session, self.serial)

        self.default_program = MFLIProgram()
        self.programs: Dict[str, MFLIProgram] = {}

        self.currently_set_program: Optional[str] = None
        self._armed_program: Optional[MFLIProgram] = None

    @classmethod
    def connect_to(cls, device_serial: str = None, **init_kwargs):
        """
        :param device_serial:     Device serial that uniquely identifies this device to the LabOne data server
        :param device_interface:  Either '1GbE' for ethernet or 'USB'
        :param data_server_addr:  Data server address. Must be already running. Default: localhost
        :param data_server_port:  Data server port. Default: 8004 for HDAWG, MF and UHF devices
        :param api_level_number:  Version of API to use for the session, higher number, newer. Default: 6 most recent
        """
        discovery = zhinst_core.ziDiscovery()
        device_id = discovery.find(device_serial)
        device_props = discovery.get(device_id)
        api_session = zhinst_core.ziDAQServer(device_props['serveraddress'], device_props['serverport'], device_props['apilevel'])
        return cls(api_session, device_props, **init_kwargs)

    def _init_daq_module(self):
        self.daq = self.api_session.dataAcquisitionModule()
        self.daq.set('device', self.serial)

    def reset(self):
        """ This function resets the device to a known default configuration.
        """
        self.read_memory.clear()

        zhinst.utils.disable_everything(self.api_session, self.serial)
        self.clear()
        self.programs.clear()

        self.reset_daq_module()

    def reset_daq_module(self):
        self.daq.finish()
        self.daq.clear()
        self._init_daq_module()

    def register_measurement_channel(self, program_name: Union[str, None] = None, window_name: str = None,
                                     channel_path: Union[str, Sequence[str]] = ()):
        """ This function saves the channel one wants to record with a certain program

        Args:
            program_name: Name of the program
            window_name: The windows for that channel.
            channel_path: the channel to record in the shape of "demods/0/sample.R.avg". Note that everything but the
            things behind the last "/" are considered to relate do the demodulator. If this is not given, you might want
             to check this driver and extend its functionality.
        """

        if isinstance(channel_path, str):
            channel_path = [channel_path]

        if program_name is None:
            program = self.default_program
        else:
            program = self.programs.setdefault(program_name, MFLIProgram())

        if window_name is None:
            program.default_channels = set(channel_path)
        else:
            if program.channel_mapping is None:
                program.channel_mapping = dict()
            program.channel_mapping[window_name] = set(channel_path)

    def register_measurement_windows(self, program_name: str, windows: Dict[str, Tuple[np.ndarray,
                                                                                       np.ndarray]]) -> None:
        """Register measurement windows for a given program. Overwrites previously defined measurement windows for
        this program.

        Args:
            program_name: Name of the program
            windows: Measurement windows by name.
                     First array are the start points of measurement windows in nanoseconds.
                     Second array are the corresponding measurement window's lengths in nanoseconds.
        """
        self.programs.setdefault(program_name, MFLIProgram()).windows = windows

        # self.programs.setdefault(program_name, {}).setdefault("windows", {}).update(windows)
        # self.programs.setdefault(program_name, {}).setdefault("windows_from_start_max", {}).update(
        #    {k: np.max(v[0] + v[1]) for k, v in windows.items()})

        # the channels we want to measure with:
        # channels_to_measure: Set[str] = self._get_channels_for_window(program_name, list(windows.keys()))
        # self.programs.setdefault(program_name, {})["all_channels"] = channels_to_measure

    # for k, v in windows.items():
    # 	self.set_measurement_mask(program_name=program_name, mask_name=k, begins=v[0], lengths=v[1])

    def register_trigger_settings(self, program_name: Union[str, None], trigger_input: Union[str, None] = None,
                                  trigger_count: int = 1,
                                  edge: Literal['rising', 'falling', 'both'] = 'rising',
                                  level: float = 0.1,
                                  delay: float = -1e-3, post_delay: float = 1e-3,
                                  measurement_count: Union[int, float] = 1,
                                  other_settings: Dict[str, Union[str, int, float, Any]] = None):
        """
        Parameters
        ----------
        program_name
            The program name to set these trigger settings for. If None is given, then these settings are used as default values.
        trigger_input
            This needs to be the path to input to the lock-in that the lock-in is able to use as a trigger (without the device serial). (see https://docs.zhinst.com/pdf/LabOneProgrammingManual.pdf for more information)
        trigger_count
            The number of trigger events to count for one measurement. This will later set the number of rows in one measurement
        edge
            The edge to look out for
        level
            the trigger level to look out for
        delay
            the delay of the start of the measurement window in relation to the time of the trigger event. Negative values will result in data being recorded before the event occurred. (in seconds)
        post_delay
            The duration to record for after the last measurement window. (in seconds)
        measurement_count
            The number of measurement to perform for one arm call. This will result in self.daq.finished() not returning true until all measurements are recorded. This will equal to trigger_count*measurement_count trigger events. The self.daq.progress() field counts the number of trigger events. If the count is set to np.inf the acquisition is not stopped after any number of triggers are received, only by calling self.stop_acquisition() or self.daq.finish(). Data, potentially incomplete and thus filled with nan, can be retrieved also in the continuous mode with the right setting of the measurement function (i.e. wait=False and fail_if_incomplete=False).
        other_settings
            Other settings to set after the standard trigger settings are send to the data server / device.
        """

        if edge not in ["rising", "falling", "both"]:
            raise ValueError(f"edge={edge} is not in ['rising', 'falling']")

        if program_name is None:
            program = self.default_program
        else:
            program = self.programs.setdefault(program_name, MFLIProgram())

        program.trigger_settings = TriggerSettings(
            trigger_input=trigger_input,
            edge=edge,
            trigger_count=trigger_count,
            level=level,
            delay=delay,
            post_delay=post_delay,
            measurement_count=measurement_count
        )
        program.other_settings = other_settings or {}
        # return
        # self.programs.setdefault(program_name, {})["trigger_settings"] = {
        #     "trigger_input": f"/{self.serial}/{trigger_input}",
        #     "edge": edge,
        #     "trigger_count": trigger_count,
        #     "level": level,
        #     "delay": delay,
        #     "post_delay": post_delay,
        #     "endless": measurement_count == np.inf,
        #     "measurement_count": 0 if measurement_count == np.inf else measurement_count,
        #     "other_settings": other_settings,
        # }

    def _get_sample_rates(self, channel: str):
        try:
            timetype_sr = TimeType().from_float(
                value=self.api_session.getDouble(f"/{self.serial}/{self._get_demod(channel)}/rate"), absolute_error=0)
            return timetype_sr
        except RuntimeError as e:
            if "ZIAPINotFoundException" in e.args[0]:
                return None
            else:
                raise

    def set_measurement_mask(self, program_name: str, mask_name: str,
                             begins: np.ndarray,
                             lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Set/overwrite a single the measurement mask for a program. Begins and lengths are in nanoseconds.

        Args:
            program_name: Name of the program
            mask_name: Name of the mask/measurement windows
            begins: Staring points in nanoseconds
            lengths: Lengths in nanoseconds

        Returns:
            Measurement windows in DAC samples (begins, lengths)
        """
        raise NotImplementedError(f"This function has been abandoned as the MFLI returns timestamps.")

    def register_operations(self, program_name: str, operations) -> None:
        """Register operations that are to be applied to the measurement results.

        Args:
            program_name: Name of the program
            operations: DAC specific instructions what to do with the data recorded by the device.
        """

        self.programs.setdefault(program_name, MFLIProgram()).operations = operations

    def _get_demod(self, channel: str):
        """ This function gets the demodulator corresponding to a channel
        """
        elements = channel.split("/")
        elements = [e for e in elements if not e.lower() == self.serial.lower()]
        elements = [e for e in elements if len(e) > 0]

        return "/".join(elements[:-1])

    def arm_program(self, program_name: str, force: Union[bool, None] = None) -> None:
        """Prepare the device for measuring the given program and wait for a trigger event."""

        force = force if force is not None else self.force_update_on_arm

        # check if program_name specified program is selected and important parameter set to the lock-in
        if self.currently_set_program is None or self.currently_set_program != program_name or force:

            self.daq.finish()
            self.daq.unsubscribe('*')

            # TODO TODO TODO TODO TODO TODO TODO TODO
            # # if the program is changed, the not returned data is removed to not have conflicts with the data parsing operations. The cleaner way would be to keep track of the time the program is changed.
            # if self.currently_set_program != program_name:
            # 	self.read()
            # 	self.daq.clear()
            # 	self._init_daq_module()
            # TODO TODO TODO TODO TODO TODO TODO TODO

            program = self.default_program.merge(self.programs[program_name])

            for c in program.required_channels():

                # activate corresponding de-modulators
                demod = self._get_demod(c)
                try:
                    self.api_session.setInt(f'/{self.serial}/{demod}/enable', 1)
                except RuntimeError as e:
                    if "ZIAPINotFoundException" in e.args[0] or f"Path /{self.serial}/{demod}/enable not found." in \
                            e.args[0]:
                        # ok, the channel can not be enabled. Then the user should be caring about that.
                        warnings.warn(
                            f"The channel {c} does not have an interface for enabling it. If needed, this can be done using the web interface.")
                        pass
                    else:
                        raise

                # select the value to measure
                self.daq.subscribe(f'/{self.serial}/{c}')

            # # check if sample rates are the same as when register_measurement_windows() was called
            # for k, v in self.programs[program_name]['masks'].items():
            # 	if len(v["channels"]) != len(v["sample_rates"]):
            # 		raise ValueError(f"There is a mismatch between number the channels to be used and the known sample rates.")
            # 	for c, r in zip(v["channels"], v["sample_rates"]):
            # 		if self._get_sample_rates(c) != r:
            # 			raise ValueError(f"The sample rate for channel '{c}' has changed. Please call register_measurement_windows() again.")

            # set the buffer size based on the largest sample rate
            # if no sample rate is readable, as for example when only AUXIN channels are used, the first demodulator is activated and the corresponding rate is used

            raw_currently_set_sample_rates: List[Union[TimeType, None]] = []
            for c in program.required_channels():
                raw_currently_set_sample_rates.append(self._get_sample_rates(c))

            logging.info(
                f"sample rates: {[(float(e) if e is not None else None) for e in raw_currently_set_sample_rates]}")

            # CAUTION
            # The MFLI lock-ins up-sample slower channels to fit the fastest sample rate.
            # This is the cased for the Lab One Data Server 21.08.20515 and the MFLi Firmware 67629.
            # TODO it needs to be verified, that this code here is actually necessary. One could also query the AUXIN using one of the demods.
            foo = [x for x in raw_currently_set_sample_rates if x is not None]
            if len(foo) == 0 and self.assumed_minimal_sample_rate is None:
                # Ok, we activate the first demodulator
                self.api_session.setInt(f'/{self.serial}/demods/0/enable', 1)
                foo.append(self._get_sample_rates(f'/{self.serial}/demods/0/sample.R'))
            if self.assumed_minimal_sample_rate is not None:
                foo.append(TimeType().from_float(value=self.assumed_minimal_sample_rate, absolute_error=0))
            max_sample_rate = max(foo)
            currently_set_sample_rates = [max_sample_rate] * len(raw_currently_set_sample_rates)

            # set daq module settings to standard things
            # TODO one might want to extend the driver to support more methods
            self.daq.set('grid/mode', 4)  # this corresponds to Mode: Exact(on-grid)
            # the following two lines set the row repetitions to 1 and off
            self.daq.set('grid/repetitions', 1)
            self.daq.set('grid/rowrepetition', 0)

            # setting trigger settings
            ts = program.trigger_settings

            rows = 1
            if ts is not None:
                rows = ts.trigger_count
                # selecting the trigger channel
                if ts.trigger_input is not None:

                    if ts.is_endless():
                        self.daq.set('endless', 1)
                    else:
                        self.daq.set('endless', 0)
                        # defines how many triggers are to be recorded in single mode i.e. endless==0
                        self.daq.set('count', ts.measurement_count)

                    if "trig" in ts.trigger_input.lower():
                        self.daq.set("type", 6)
                    else:
                        self.daq.set("type", 1)

                    self.daq.set("triggernode", f"/{self.serial}/{ts.trigger_input}")

                    edge_key = ["rising", "falling", "both"].index(ts.edge)
                    self.daq.set("edge", edge_key)

                    if "trigin" in ts.trigger_input.lower():
                        _trigger_id = int(ts.trigger_input.split("TrigIn")[-1])
                        assert _trigger_id in [1, 2]
                        self.api_session.setDouble(f'/{self.serial}/triggers/in/{_trigger_id-1}/level', ts.level);
                    else:
                        self.daq.set("level", ts.level)

                    self.daq.set("delay", ts.delay)
                    self.daq.set('bandwidth', 0)

                else:
                    self.daq.set("type", 0)

                self.daq.set('count', rows)

            if program.other_settings:
                for k, v in program.other_settings.items():
                    self.daq.set(k, v)

            # set the buffer size according to the largest measurement window
            # TODO one might be able to implement this a bit more cleverly
            measurement_duration = self.programs[program_name].get_minimal_duration()
            measurement_duration += (ts.post_delay + -1 * ts.delay) * 1e9
            larges_number_of_samples = max_sample_rate / 10 ** 9 * measurement_duration
            larges_number_of_samples = np.ceil(larges_number_of_samples)
            self.daq.set('grid/cols', larges_number_of_samples)
            self.daq.set('grid/rows', rows)  # this corresponds to measuring only for one trigger

            # self.daq.set("buffersize", 2*measurement_duration) # that the buffer size is set to be larger than the duration is something that the SM script did.
            # # --> in the current version and/or configuration, this path is read-only.

            self.currently_set_program = program_name

            logging.info(
                f"Will record {larges_number_of_samples} samples per row for {measurement_duration * 1e-9}s!")  # TODO this will have to change if proper multi triggers with over multiple rows is going to be used.
            logging.info(f"{rows} row(s) will be recorded.")
            logging.info(f"the following trigger settings will be used: {ts}")
            logging.info(f"MFLI returns a duration of {self.daq.get('duration')['duration'][0]}s")

            self._armed_program = program

        # execute daq
        self.daq.execute()

        # wait until changes have taken place
        self.api_session.sync()

    def unarm_program(self):
        """ unarms the lock-in. This should be program independent.
        """

        self.daq.finish()
        self.daq.unsubscribe('*')
        self.api_session.sync()

        self.currently_set_program = None

    def force_trigger(self, *args, **kwargs):
        """ forces a trigger event
        """
        self.daq.set('forcetrigger', 1)

    def stop_acquisition(self):
        self.daq.finish()

    def clear_memory(self):
        self.read_memory.clear()

    def delete_program(self, program_name: str) -> None:
        """Delete program from internal memory."""

        # this does not have an effect on the current implementation of the lock-in driver.

        if self.currently_set_program == program_name:
            self.unarm_program()

        self.programs.pop(program_name)

    def clear(self) -> None:
        """Clears all registered programs."""
        self.unarm_program()
        self.read_memory.clear()

    def get_mfli_data(self,
                      wait: bool = True,
                      timeout: float = np.inf,
                      wait_time: float = 1e-3,
                      return_raw: bool = False,
                      fail_if_incomplete: bool = False,
                      fail_on_empty: bool = False):
        """Get the last measurement's results of the specified channels

                Parameters
                ----------
                wait: bool, optional
                    Should the code wait until the acquisition has finished? Else incomplete data might be returned. (default: True)
                timeout: float, optional
                    The time to wait until the measurement is stopped in units of seconds. (default: np.inf)
                wait_time : float, optional
                    The time to pause in the until querying the data server for new data. (default = 1e-3 = 1ms)
                return_raw: bool, optional
                    If True, the function will return the raw data without selecting the measurement windows. This will then be in the shape of data[channel_name][shot_index]: xr.DataArray.
                    If False, the return value will have the structure data[window_name][channel_name][mask_index]: xr.DataArray.
                    Also, if False, the time axis will be shifted, such that the trigger occurred at data[window_name][channel_name][mask_index]["time"]==0 #ns.
                fail_if_incomplete: bool, optional
                    if True and the timeout has been reached and the acquisition has not finished, an error will be raised.
                fail_on_empty:bool, optional
                    if one of the channels is empty, which occurred in the development process for large sample rates, an error is raised in the parsing function and this these incomplete measurements will sit in this classes memory, never to be returned. If False, the empty channels are ignored and it will be returned, what every is there.

                Note
                ----
                - There is currently no mechanism implemented to keep track of the relation between received data and underlying program (i.e. measurement windows, ...). This has to be tracked by the user!
                - When the parameter wait=False and fail_if_incomplete=False are given, then incomplete data is parsed and returned. This can lied to receiving the same measurement multiple times. The user has to keep track of that!

                """

        program = self._armed_program

        if callable(program.operations):
            program.operations = [program.operations]

        if program.operations is None or len(program.operations) == 0:
            return_raw = True

        # wait until the data acquisition has finished
        # TODO implement timeout
        _endless_flag_helper = program.trigger_settings.is_endless()

        start_waiting = time.time()
        if not self.daq.finished() and wait:
            logging.info(f"Waiting for device {self.serial} to finish the acquisition...")
            logging.info(f"Progress: {self.daq.progress()[0]}")
            while not self.daq.finished() and wait and not (
                    time.time() - start_waiting > timeout) and not _endless_flag_helper:
                time.sleep(wait_time)

        if fail_if_incomplete and not self.daq.finished():
            raise ValueError(f"Device {self.serial} did not finish the acquisition in time.")

        data = self.daq.read()
        self.daq_read_return.update(data)

        if data is None or len(data) == 0:
            warnings.warn(f"Reading form the data acquisition module did not work.")

        clockbase = self.api_session.getDouble(f'/{self.serial}/clockbase')

        # go through the returned object and extract the data of interest

        recorded_data = {}

        for device_name, device_data in data.items():
            if device_name == self.serial.lower():
                for input_name, input_data in device_data.items():
                    for signal_name, signal_data in input_data.items():
                        for final_level_name, final_level_data in signal_data.items():
                            channel_name = f"/{device_name}/{input_name}/{signal_name}/{final_level_name}".lower()
                            channel_data = []
                            for i, d in enumerate(final_level_data):
                                converted_timestamps = {
                                    "systemtime_converted": d['header']["systemtime"] / clockbase * 1e9,
                                    "createdtimestamp_converted": d['header'][
                                                                      "createdtimestamp"] / clockbase * 1e9,
                                    "changedtimestamp_converted": d['header'][
                                                                      "changedtimestamp"] / clockbase * 1e9,
                                }
                                channel_data.append(xr.DataArray(
                                    data=d["value"],
                                    coords={'time': (['row', 'col'], d["timestamp"] / clockbase * 1e9)},
                                    dims=['row', 'col'],
                                    name=channel_name,
                                    attrs={**d['header'], **converted_timestamps, "device_serial": self.serial,
                                           "channel_name": channel_name}))
                            recorded_data[channel_name] = channel_data

        # check if the shapes of the received measurements are the same.
        # this is needed as the assumption, that the lock-in/data server up-samples slower channels to match the one with the highest rate.

        recorded_shapes = {k: set([e.shape for e in v]) for k, v in recorded_data.items()}
        if any([len(v) > 1 for v in recorded_shapes.values()]) or len(
                set([e for a in recorded_shapes.values() for e in a])) > 1:
            warnings.warn(
                f"For at least one received channel entries with different dimensions are present. This might lead to undesired masking! (The code will not raise an exception.) ({recorded_shapes})")

        if len(recorded_data) == 0:
            warnings.warn(f"No data has been recorded!")

        # update measurements in local memory
        for k, v in recorded_data.items():
            self.read_memory.setdefault(k, [])
            # for all the measurement that we just read of the device:
            for m in v:
                # get the time stamp of when the measurement was created
                crts = m.attrs["createdtimestamp"][0]

                # now look, if that measurement is already in the memory
                for i, e in enumerate(self.read_memory[k]):
                    if e.attrs["createdtimestamp"][0] == crts:
                        # then we can overwrite that.
                        # TODO don't overwrite that. Only replace nan values
                        self.read_memory[k][i] = m
                        break
                else:
                    # if we did not find that element in the list, we append it.
                    self.read_memory[k].append(m)

            # sort the element by their createdtimestamp
            order = np.argsort([e.attrs["createdtimestamp"][0] for e in self.read_memory[k]])
            self.read_memory[k] = [self.read_memory[k][o] for o in order]
        # CAUTION this only sorts the ones that have been updated. This might not be intended!!!

        if return_raw:
            return recorded_data

        # now we package everything in self.read_memory, such that the elements with one creation time stamp are processed at once.
        # If for every self._get_channels_for_window(self.currently_set_program, None) some measurement is present: try parsing the data. If that was successful: remove these elements from the read_memeory and return (or yield) the results.

        # TODO this might lead to leaving some measurements in the memory that will never be used, if they don't get related with measurements from the other channels of the program.

        creation_ts = {k: [e.attrs["createdtimestamp"][0] for e in v] for k, v in self.read_memory.items()}
        if len(creation_ts) == 0:
            # Then we have nothing to process
            return None
        all_ts = np.unique(np.concatenate(list(creation_ts.values())))
        assert len(all_ts.shape) == 1
        if len(all_ts) == 0:
            # Then we have nothing to process
            return None

        channels_to_measure = []
        for k, v in program.windows.items():
            channels_to_measure.extend(program.channel_mapping[k])
        channels_to_measure = list(np.unique(channels_to_measure))
        channels_to_measure = [f"/{self.serial}/{c}".lower() for c in channels_to_measure]

        things_to_remove = {}

        results = []

        for ts in all_ts:
            contained = [k for k, v in creation_ts.items() if ts in v]
            if all([(c.lower() in contained) for c in channels_to_measure]):
                # then we have all measurement for that shot
                that_shot = {}
                _indexs = [creation_ts[c].index(ts) for c in channels_to_measure]

                for c, i in zip(channels_to_measure, _indexs):
                    that_shot[c] = [self.read_memory[c][i]]
                    # Here the inner list could be removed if one would remove the old functionality from the _parse_data function.

                _the_warning_string = f"Parsing some data did not work. This might fix itself later, when the missing data is retrieved from the device. If not, clearing the memory (i.e. self.clear_memory()), resetting the daq_module (i.e. self.reset_daq_module()), or setting the field to the selected program (i.e., self.currently_set_program=None) to None and then rearming the original program might work. For debugging purposes, one might want to call the measure function with the return_raw=True parameter."
                try:
                    that_shot_parsed = program.operations[0](serial=self.serial, recorded_data=that_shot, program=program, fail_on_empty=fail_on_empty)
                except IndexError as e:
                    traceback.print_exc()
                    warnings.warn(_the_warning_string)
                except KeyError as e:
                    traceback.print_exc()
                    warnings.warn(_the_warning_string)
                except ValueError as e:
                    if "The received data for channel" in str(e):
                        pass
                    else:
                        raise
                else:
                    # the parsing worked, we can now remove the data from the memory
                    results.append(that_shot_parsed)
                    for c, i in zip(channels_to_measure, _indexs):
                        things_to_remove.setdefault(c, []).append(i)

            else:
                pass
            # TODO do something here. Maybe raise a warning.

        # then we can remove the element that worked:
        for k, v in things_to_remove.items():
            v.sort()
            assert (len(v) == 0) or (v[0] <= v[-1])
            for i in reversed(v):
                self.read_memory[k].pop(i)

        if not return_raw:
            if len(results) == 0:
                raise ValueError()
                return None
            return results

    def measure_program(self,
                        channels: Iterable[str] = None,
                        wait: bool = True,
                        timeout: float = np.inf,
                        wait_time: float = 1e-3,
                        return_raw: bool = False,
                        fail_if_incomplete: bool = False,
                        fail_on_empty: bool = False) -> Union[Dict[str, List[xr.DataArray]],
                                                              Dict[str, Dict[str, List[xr.DataArray]]],
                                                              None]:
        return self.get_mfli_data(wait, timeout, wait_time, return_raw, fail_if_incomplete, fail_on_empty)
