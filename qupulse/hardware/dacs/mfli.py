""" This file contains a qupulse driver for the DAQ module of an Zuerich Instruments MFLI.

May lines of code have been adapted from the zihdawg driver.

"""
from typing import Dict, Tuple, Iterable, Union, List, Set, Any
from enum import Enum
import warnings
import time
import traceback
import xarray as xr
import numpy as np
import numpy.typing as npt

from qupulse.utils.types import TimeType
from qupulse.hardware.dacs.dac_base import DAC

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



"""

  TODO  
========
[X] make TriggerMode class an Enum
[X] setup minimal connection without changing settings other than buffer lengths
[X] extract window things
[X] rethink handling different sample rates!
[X] cut obtained data to fit into requested windows
[X] provide interface for changing trigger settings
[X] print information about how long the measurement is expected to run
[X] implement multiple triggers (using rows) (and check how this actually behaves)
	[X] count
	[X] endless
		[X] check how that would behave. Does that overwrite or shift things?
	[X] change in trigger input port
[X] implement setting recording channel (could that be already something inside qupulse?)
[ ] see why for high sample rates (e.g. 857.1k) things crash or don't behave as expected
[ ] Implement yield for not picked up data (read() was not called)
=> this should be sufficient for operation
[ ] implement optional operations (averaging, binning, up/down sampling, ...)
[ ] implement scope interface for higher sample rates (if i understood the documentation correctly)
[ ] implement low level interface (subscribe())


Tests to implement:
[ ] connecting to a MFLI Device by querying
	[ ] available nodes of the api_session
	[ ] creating and reading a DAQ Module
[ ] registering channels
	[ ] Demod 0 and 1
	[ ] AUXIN
[ ] defining measurement windows
	[ ] consecutive ones
	[ ] overlapping
	[ ] some windows with names but no begin and length information
[ ] relating channels to windows
	[ ] adding one channel to one window
	[ ] adding multiple channels to one windows
	[ ] adding one channel to multiple windows
[ ] measuring only one AUXIN (some channel without the rate argument)
[ ] finishing the acquisition before all rows are recorded (data processing (throwing out nans) might not work as intended)
[ ] what happens when the lock-in is not returning what it should?
	[ ] missing channels
	[ ] channels separated over multiple read() calls



"""


class TriggerType(Enum):

	# https://docs.zhinst.com/pdf/LabOneProgrammingManual.pdf
	# | Mode / Trigger Type | Description | Value of type |
	# |---------------------|-------------|---------------|
	# | Continuous | Continuous recording of data. | 0 |
	# | Edge | Edge trigger with noise rejection. | 1 |
	# | Pulse | Pulse width trigger with noise rejection. | 3 |
	# | Tracking (Edge or Pulse) | Level tracking trigger to compensate for signal drift. | 4 |
	# | Digital | Digital trigger with bit masking. | 2 |
	# | Hardware | Trigger on one of the instrument’s hardware trigger channels (not available on HF2). | 6 |
	# |Pulse Counter | Trigger on the value of an instrument’s pulse counter (requires CNT Option). | 8 |

	CONTINUOUS = 0
	EDGE = 1
	DIGITAL = 2
	PULSE = 3
	TRACKING = 4
	HARDWARE = 6
	PULSE_COUNTER = 8



class MFLIDAQ(DAC):
	""" This class contains the driver for using the DAQ module of an Zuerich Instruments MFLI with qupulse.
	"""
	def __init__(self, device_serial: str = None,
				device_interface: str = '1GbE',
				data_server_addr: str = 'localhost',
				data_server_port: int = 8004,
				api_level_number: int = 6,
				reset: bool = False,
				timeout: float = 20) -> None:
		"""
		:param device_serial:     Device serial that uniquely identifies this device to the LabOne data server
		:param device_interface:  Either '1GbE' for ethernet or 'USB'
		:param data_server_addr:  Data server address. Must be already running. Default: localhost
		:param data_server_port:  Data server port. Default: 8004 for HDAWG, MF and UHF devices
		:param api_level_number:  Version of API to use for the session, higher number, newer. Default: 6 most recent
		:param reset:             Reset device before initialization
		:param timeout:           Timeout in seconds for uploading
		"""
		self.api_session = zhinst_core.ziDAQServer(data_server_addr, data_server_port, api_level_number)
		# assert zhinst.utils.api_server_version_check(self.api_session)  # Check equal data server and api version.
		self.device_interface = device_interface
		self.device = self.api_session.connectDevice(device_serial, device_interface)
		self.default_timeout = timeout
		self.serial = device_serial

		self.daq = None
		self._init_daq_module()

		self.force_update_on_arm:bool = True
		self.assumed_minimal_sample_rate:Union[float, None] = None # in units of Sa/s

		self.daq_read_return = {}
		self.read_memory = {} # self.read_memory[<path/channel>]:List[xr.DataArray]

		if reset:
			# Create a base configuration: Disable all available outputs, awgs, demods, scopes,...
			zhinst.utils.disable_everything(self.api_session, self.serial)

		self.programs = {}
		self.currently_set_program = None

	def _init_daq_module(self):
		self.daq = self.api_session.dataAcquisitionModule()
		self.daq.set('device', self.serial)
	
	def reset(self):
		""" This function resets the device to a known default configuration.
		"""
		self.read_memory = {}

		zhinst.utils.disable_everything(self.api_session, self.serial)
		self.clear()

		self.reset_daq_module()

	def reset_daq_module(self):
		self.daq.finish()
		self.daq.clear()
		self._init_daq_module()

	def register_measurement_channel(self, program_name:Union[str, None]=None, channel_path:Union[str, List[str]]=[], window_name:str=None):
		""" This function saves the channel one wants to record with a certain program

		Args:
			program_name: Name of the program
			channel_path: the channel to record in the shape of "demods/0/sample.R.avg". Note that everything but the things behind the last "/" are considered to relate do the demodulator. If this is not given, you might want to check this driver and extend its functionality.
			window_name: The windows for that channel.

		"""

		if not isinstance(channel_path, list):
			channel_path = [channel_path]
		self.programs.setdefault(program_name, {}).setdefault("channel_mapping", {}).setdefault(window_name, set()).update(channel_path)

	def register_measurement_windows(self, program_name:str, windows: Dict[str, Tuple[np.ndarray,
																					   np.ndarray]]) -> None:
		"""Register measurement windows for a given program. Overwrites previously defined measurement windows for
		this program.

		Args:
			program_name: Name of the program
			windows: Measurement windows by name.
					 First array are the start points of measurement windows in nanoseconds.
					 Second array are the corresponding measurement window's lengths in nanoseconds.
		"""

		self.programs.setdefault(program_name, {}).setdefault("windows", {}).update(windows)
		self.programs.setdefault(program_name, {}).setdefault("windows_from_start_max", {}).update({k:np.max(v[0]+v[1]) for k, v in windows.items()})

		# the channels we want to measure with:
		channels_to_measure: Set[str] = self._get_channels_for_window(program_name, list(windows.keys()))
		self.programs.setdefault(program_name, {})["all_channels"] = channels_to_measure

		# for k, v in windows.items():
		# 	self.set_measurement_mask(program_name=program_name, mask_name=k, begins=v[0], lengths=v[1])

	def register_trigger_settings(self, program_name:str, trigger_input:Union[str, None]=None, trigger_count:int=1, edge:str='rising', level:float=0.1, delay:float=-1e-3, post_delay:float=1e-3, measurement_count:Union[int, float]=1, other_settings:Dict[str, Union[str, int, float, Any]]={}):
		"""
		Parameters
		----------
		program_name
			The program name to set these trigger settings for.
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

		self.programs.setdefault(program_name, {})["trigger_settings"] = {
			"trigger_input": f"/{self.serial}/{trigger_input}",
			"edge": edge,
			"trigger_count": trigger_count,
			"level": level,
			"delay": delay,
			"post_delay": post_delay,
			"endless": measurement_count==np.inf,
			"measurement_count": 0 if measurement_count==np.inf else measurement_count,
			"other_settings": other_settings,
		}

	def _get_sample_rates(self, channel:str):
		try:
			timetype_sr = TimeType().from_float(value=self.api_session.getDouble(f"/{self.serial}/{self._get_demod(channel)}/rate"), absolute_error=0)
			return timetype_sr
		except RuntimeError  as e:
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

		assert begins.shape == lengths.shape

		if program_name not in self.programs:
			raise ValueError(f"Program '{program_name}' not known")

		# the channels we want to measure with:
		channels_to_measure: List[str] = self._get_channels_for_window(program_name, mask_name)

		if len(channels_to_measure) == 0:
			warnings.warn(f"There are no channels defined that should be measured in mask '{mask_name}'.")

		# get the sample rates for the requested channels. If no sample rate is found, None will be used. This code is not very nice.
		raw_currently_set_sample_rates: List[Union[TimeType, None]] = []

		for c in channels_to_measure:
			raw_currently_set_sample_rates.append(self._get_sample_rates(c))

		# CAUTION
		# The MFLI lock-ins up-sample slower channels to fit the fastest sample rate.
		# This is the cased for the Lab One Data Server 21.08.20515 and the MFLi Firmware 67629.
		foo = [x for x in raw_currently_set_sample_rates if x is not None]
		if len(foo) == 0 and self.assumed_minimal_sample_rate is None:
			raise ValueError(f"No information about the sample rate is given, thus we can not calculate the window sizes.")
		if self.assumed_minimal_sample_rate is not None:
			foo.append(TimeType().from_float(value=self.assumed_minimal_sample_rate, absolute_error=0))
		max_sample_rate = max(foo)
		currently_set_sample_rates = [max_sample_rate]*len(raw_currently_set_sample_rates)

		mask_info = np.full((3, len(begins), len(currently_set_sample_rates)), np.nan)

		for i, _sr in enumerate(currently_set_sample_rates):
			if _sr is not None:
				sr = _sr*1e-9 # converting the sample rate, which is given in Sa/s, into Sa/ns
				# this code was taken from the already implemented alazar driver. 
				mask_info[0, :, i] = np.rint(begins * float(sr)).astype(dtype=np.uint64) # the begin
				mask_info[1, :, i] = np.floor_divide(lengths * float(sr.numerator), float(sr.denominator)).astype(dtype=np.uint64) # the length
				mask_info[2, :, i] = (mask_info[0, :, i] + mask_info[1, :, i]).astype(dtype=np.uint64) # the end

		self.programs.setdefault(program_name, {}).setdefault("masks", {})[mask_name] = {"mask": mask_info, "channels": channels_to_measure, "sample_rates": raw_currently_set_sample_rates}
		self.programs.setdefault(program_name, {}).setdefault("all_channels", set()).update(channels_to_measure)
		self.programs.setdefault(program_name, {}).setdefault("window_hull", [np.nan, np.nan])
		# self.programs.setdefault(program_name, {}).setdefault("largest_sample_rate", -1*np.inf) # This will be used to set the window_hull only based on the fastest channel queried. 

		# as the lock-in can measure multiple channels with different sample rates, the return value of this function is not defined correctly.
		# this could be fixed by only measuring on one channel, or by returning some "summary" value. As of now, there does not to be a use of the return values.
		# but there are also used calculations in the following code.

		if len(channels_to_measure) == 0:
			return None
		else:

			# update the hull of all measurement windows defined for this program to later set the number of samples to record. 
			# TODO this whole thing could be improved at some point.
			# we also only use the max sample value later. The smallest staring point is somewhat ill-defined
			if np.sum(np.isnan(mask_info)) == len(mask_info.reshape((-1))):
				pass
				print(f"will not use mask {mask_name}")
			else:
				# TODO need to do something about the different sample rates!!!!
				# maybe have it not this flexible???

				_start = np.nanmin(mask_info[0])
				_end = np.nanmax(mask_info[2])
				if np.isnan(self.programs[program_name]["window_hull"][0]) or self.programs[program_name]["window_hull"][0] > _start:
					self.programs[program_name]["window_hull"][0] = _start
				if np.isnan(self.programs[program_name]["window_hull"][1]) or self.programs[program_name]["window_hull"][1] < _end:
					self.programs[program_name]["window_hull"][1] = _end

			return (np.min(mask_info[0], axis=-1), np.max(mask_info[2], axis=-1))

	def register_operations(self, program_name:str, operations) -> None:
		"""Register operations that are to be applied to the measurement results.

		Args:
			program_name: Name of the program
			operations: DAC specific instructions what to do with the data recorded by the device.
		"""

		self.programs.setdefault(program_name, {}).setdefault("operations", []).append(operations)

	def _get_channels_for_window(self, program_name:Union[str, None], window_name:Union[str, List[str], None]=None):
		""" Returns the channels to be measured for a given window
		"""
		if window_name is None:
			window_name = list(self.programs[program_name]["windows"].keys())
		if not isinstance(window_name, list):
			window_name = [window_name]

		channels: Set[str] = set()

		for wn in window_name:
			try:
				channels.update(self.programs[program_name]["channel_mapping"][wn])
			except KeyError:
				try:
					channels.update(self.programs[program_name]["channel_mapping"][None])
				except KeyError:
					try:
						channels.update(self.programs[None]["channel_mapping"][wn])
					except KeyError:
						try:
							channels.update(self.programs[None]["channel_mapping"][None])
						except KeyError:
							pass

		if len(channels) == 0:
			warnings.warn(f"No channels registered to measure with program {program_name} and window {window_name}.")

		channels = set([e.lower() for e in channels])
		return channels
	
	def _get_demod(self, channel:str):
		""" This function gets the demodulator corresponding to a channel
		"""
		elements = channel.split("/")
		elements = [e for e in elements if not e.lower()==self.serial.lower()]
		elements = [e for e in elements if len(e) > 0]

		return "/".join(elements[:-1])

	def arm_program(self, program_name: str, force:Union[bool, None]=None) -> None:
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

			for c in self.programs[program_name]["all_channels"]:

				# activate corresponding demodulators
				demod = self._get_demod(c)
				try:
					self.api_session.setInt(f'/{self.serial}/{demod}/enable', 1)
				except RuntimeError as e:
					if "ZIAPINotFoundException" in e.args[0] or f"Path /{self.serial}/{demod}/enable not found." in e.args[0]:
						# ok, the channel can not be enabled. Then the user should be caring about that.
						warnings.warn(f"The channel {c} does not have an interface for enabling it. If needed, this can be done using the web interface.")
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
			for c in self.programs[program_name]["all_channels"]:
				raw_currently_set_sample_rates.append(self._get_sample_rates(c))

			print(f"sample rates: {[(float(e) if e is not None else None) for e in raw_currently_set_sample_rates]}")

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
			currently_set_sample_rates = [max_sample_rate]*len(raw_currently_set_sample_rates)


			# set daq module settings to standard things
			# TODO one might want to extend the driver to support more methods
			self.daq.set('grid/mode', 4) # this corresponds to Mode: Exact(on-grid)
			# the following two lines set the row repetitions to 1 and off
			self.daq.set('grid/repetitions', 1)
			self.daq.set('grid/rowrepetition', 0)

			# setting trigger settings
			ts = {}
			try:
				ts.update(self.programs[None]["trigger_settings"])
			except KeyError:
				pass
			try:
				ts.update(self.programs[program_name]["trigger_settings"])
			except KeyError:
				pass

			rows = 1
			if len(ts) != 0:
				rows = ts["trigger_count"]
				# selecting the trigger channel
				if ts["trigger_input"] is not None:

					if ts["endless"]:
						self.daq.set('endless', 1)
					else:
						self.daq.set('endless', 0)
						self.daq.set('count', ts["measurement_count"]) # defines how many triggers are to be recorded in single mode i.e. endless==0

					if "trig" in ts["trigger_input"].lower():
						self.daq.set("type", 6)
					else:
						self.daq.set("type", 1)

					self.daq.set("triggernode", ts["trigger_input"])

					edge_key = ["rising", "falling", "both"].index(ts["edge"])
					self.daq.set("edge", edge_key)
					
					self.daq.set("level", ts["level"])


					self.daq.set("delay", ts["delay"])
					self.daq.set('bandwidth', 0)

				else:
					self.daq.set("type", 0)

				self.daq.set('count', rows)

				for k, v in ts["other_settings"].items():
					self.daq.set(k, v)



			# set the buffer size according to the largest measurement window
			# TODO one might be able to implement this a bit more cleverly
			measurement_duration = np.max(list(self.programs[program_name]["windows_from_start_max"].values()))
			measurement_duration += ts["post_delay"]*1e-9
			larges_number_of_samples = 1e-9*max_sample_rate*measurement_duration
			larges_number_of_samples = np.ceil(larges_number_of_samples)
			self.daq.set('grid/cols', larges_number_of_samples)
			self.daq.set('grid/rows', rows) # this corresponds to measuring only for one trigger

			# self.daq.set("buffersize", 2*measurement_duration) # that the buffer size is set to be larger than the duration is something that the SM script did. 
			# # --> in the current version and/or configuration, this path is read-only.

			self.currently_set_program = program_name

			print(f"Will record {larges_number_of_samples} per row samples for {measurement_duration*1e-9}s!") # TODO this will have to change if proper multi triggers with over multiple rows is going to be used.
			print(f"{rows} row(s) will be recorded.")
			print(f"the following trigger settings will be used: {ts}")
			print(f"MFLI returns a duration of {self.daq.get('duration')['duration'][0]}s")

		# execute daq
		self.daq.execute()

		# wait until changes have taken place
		self.api_session.sync()

	def unarm_program(self, program_name:str):
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
		self.read_memory = {}

	def delete_program(self, program_name: str) -> None:
		"""Delete program from internal memory."""

		# this does not have an effect on the current implementation of the lock-in driver.

		if self.currently_set_program == program_name:
			self.unarm_program(program_name)

		self.programs.pop(program_name)

	def clear(self) -> None:
		"""Clears all registered programs."""

		self.unarm_program(program_name=None)
		self.read_memory = {}

	def _parse_data(self, recorded_data, program_name:str, fail_on_empty:bool=True):
		""" This function parses the recorded data and extracts the measurement masks and applies optional operations
		"""

		# the first dimension of channel_data is expected to be the history of multiple not read data points. This will be handled as multiple entries in a list. This will then not make too much sense, if not every channel as this many entries. If this is the case, they will be stacked, such that for the last elements it fits.
		# TODO do this based on the timestamps and not the indices. That might be more sound than just assuming that.


		# applying measurement windows and optional operations
		# TODO implement operations

		# targeted structure:
		# results[<mask_name>][<channel>] -> [data]

		masked_data = {}

		# the MFLI returns a list of measurements. We only proceed with the last ones from this list. One might want to iterate over that and process all of them. 
		# This feature might be useful if after some measurements no read() operation is called. Then with the later read, the data is returned.
		# TODO this might be more elegantly implemented or handled using yields!
		shot_index = 0 # TODO make this more flexible to not lose things

		for window_name in self.programs[program_name]["windows"]:
			data_by_channel = {}
			_wind = self.programs[program_name]["windows"][window_name]
			for ci, _cn in enumerate(self._get_channels_for_window(program_name, window_name)):
				cn = f"/{self.serial}/{_cn}".lower()
				# print(cn)
				if len(recorded_data[cn]) <= shot_index:
					# then we do not have data for this shot_index, which is intended to cover multiple not yet collected measurements. And thus will not have anything to save.
					warnings.warn(f"for channel '{cn}' only {len(recorded_data[cn])} shots are given. This does not allow for taking element [-1-{shot_index}]")
					continue
				applicable_data = recorded_data[cn][-1-shot_index]
				applicable_data = applicable_data.where(~np.isnan(applicable_data), drop=True)

				if len(applicable_data)==0 or np.product([*applicable_data.shape])==0:
					if fail_on_empty:
						raise ValueError(f"The received data for channel {_cn} is empty.")
					else:
						warnings.warn(f"The received data for channel {_cn} is empty.")
						continue

				extracted_data = []
				for b, l in zip(*_wind):
					# _time_of_first_not_nan_value = applicable_data.where(~np.isnan(applicable_data), drop=True)["time"][:, 0].values

					_time_of_first_not_nan_value = applicable_data["time"][:, 0].values

					time_of_trigger = applicable_data.attrs["gridcoloffset"][0]*1e9+_time_of_first_not_nan_value

					# print(f"time_of_trigger={time_of_trigger}")
					foo = applicable_data.where((applicable_data["time"]>=(time_of_trigger+b)[:, None]) & (applicable_data["time"]<=(time_of_trigger+b+l)[:, None]), drop=False).copy()
					foo2 = foo.where(~np.isnan(foo), drop=True)
					rows_with_data = np.sum(~np.isnan(foo), axis=-1)>0
					foo2["time"] -= time_of_trigger[rows_with_data, None]
					extracted_data.append(foo2)

				# print(f"extracted_data={extracted_data}")

				data_by_channel.update({cn: extracted_data})
			masked_data[window_name] = data_by_channel

		return masked_data

	def measure_program(self, channels: Iterable[str] = [], wait:bool=True, timeout:float=np.inf, return_raw:bool=False, fail_if_incomplete:bool=False, fail_on_empty:bool=False, program_name:Union[str, None]=None) -> Union[Dict[str, List[xr.DataArray]], Dict[str, Dict[str, List[xr.DataArray]]]]:
		"""Get the last measurement's results of the specified operations/channels
		
		Parameters
		----------
		channels: Iterable[str], optional
			Has no function here. (default: [])
		wait: bool, optional
			Should the code wait until the acquisition has finished? Else incomplete data might be returned. (default: True)
		timeout: float, optional
			The time to wait until the measurement is stopped in units of seconds. (default: np.inf)
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

		program_name = program_name if program_name is not None else self.currently_set_program

		# wait until the data acquisition has finished
		# TODO implement timeout
		start_waiting = time.time()
		while not self.daq.finished() and wait and not (time.time()-start_waiting>timeout) and not self.programs[program_name]["trigger_settings"]["endless"]:
			time.sleep(1)
			print(f"Waiting for device {self.serial} to finish the acquisition...") 
			print(f"Progress: {self.daq.progress()[0]}")

		if fail_if_incomplete and not self.daq.finished():
			raise ValueError(f"Device {self.serial} did not finish the acquisition in time.")

		data = self.daq.read()
		self.daq_read_return.update(data)

		self.clockbase = self.api_session.getDouble(f'/{self.serial}/clockbase')

		# go through the returned object and extract the data of interest

		recorded_data = {}

		for device_name, device_data in data.items():
			if device_name == self.serial:
				for input_name, input_data in device_data.items():
					for signal_name, signal_data in input_data.items():
						for final_level_name, final_level_data in signal_data.items():
							channel_name = f"/{device_name}/{input_name}/{signal_name}/{final_level_name}".lower()
							channel_data = []
							for i, d in enumerate(final_level_data):
								converted_timestamps = {
									"systemtime_converted": d["systemtime"]/self.clockbase*1e9,
									"createdtimestamp_converted": d["createdtimestamp"]/self.clockbase*1e9,
									"changedtimestamp_converted": d["changedtimestamp"]/self.clockbase*1e9,
								}
								channel_data.append(xr.DataArray(
											data=d["value"],
											coords={'time': (['row', 'col'], d["timestamp"]/self.clockbase*1e9)},
											dims=['row', 'col'],
											name=channel_name,
											attrs={**d['header'], **converted_timestamps, "device_serial": self.serial, "channel_name": channel_name}))
							recorded_data[channel_name] = channel_data

		# check if the shapes of the received measurements are the same. 
		# this is needed as the assumption, that the lock-in/data server up-samples slower channels to match the one with the highest rate.

		recorded_shapes = {k:set([e.shape for e in v]) for k, v in recorded_data.items()}
		if any([len(v)>1 for v in recorded_shapes.values()]) or len(set([e for a in recorded_shapes.values() for e in a]))>1:
			warnings.warn(f"For at least one received channel entries with different dimensions are present. This might lead to undesired masking! (The code will not raise an exception.) ({recorded_shapes})")

		if len(recorded_data) == 0:
			warnings.warn(f"No data has been recorded!")

		# update measurements in local memory
		for k, v in recorded_data.items():
			self.read_memory.setdefault(k, [])
			# for all the measurement that we just read of the device:
			for m in v:
				# get the time stamp of when the measurement was created
				crts = m.attrs["createdtimestamp"][0]
				
				# now look, if that measurement is measurement is already in the memory
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

		creation_ts = {k:[e.attrs["createdtimestamp"][0] for e in v] for k, v in self.read_memory.items()}
		if len(creation_ts)==0:
			# Then we have nothing to process
			return None
		all_ts = np.unique(np.concatenate(list(creation_ts.values())))
		assert len(all_ts.shape) == 1
		if len(all_ts)==0:
			# Then we have nothing to process
			return None

		channels_to_measure = self._get_channels_for_window(self.currently_set_program, None)
		channels_to_measure = [f"/{self.serial}/{c}" for c in channels_to_measure]

		things_to_remove = {}

		results = []

		for ts in all_ts:
			contained = [k for k, v in creation_ts.items() if ts in v]
			if all([(c in contained) for c in channels_to_measure]):
				# then we have all measurement for that shot
				that_shot = {}
				_indexs = [creation_ts[c].index(ts) for c in channels_to_measure]
				
				for c, i in zip(channels_to_measure, _indexs):
					that_shot[c] = [self.read_memory[c][i]] # Here the inner list could be removed if one would remove the old functionality from the _parse_data function.

				_the_warning_string = f"Parsing some data did not work. This might fix itself later, when the missing data is retrieved from the device. If not, clearing the memory (i.e. self.clear_memory()), resetting the daq_module (i.e. self.reset_daq_module()), or setting the field to the selected program (i.e., self.currently_set_program=None) to None and then rearming the original program might work. For debugging purposes, one might want to call the measure function with the return_raw=True parameter."
				try:
					that_shot_parsed = self._parse_data(that_shot, self.currently_set_program, fail_on_empty=fail_on_empty)
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
			assert (len(v)==0) or (v[0] <= v[-1])
			for i in reversed(v):
				self.read_memory[k].pop(i)

		if not return_raw:
			if len(results) == 0:
				return None
			return results