""" This file contains a qupulse driver for the DAQ module of an Zuerich Instruments MFLI.

May lines of code have been adapted from the zihdawg driver.

"""
from typing import Dict, Tuple, Iterable, Union, List
from enum import Enum
import warnings
import time
import xarray as xr

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

# from zhinst.toolkit import Session
import numpy as np


"""

  TODO  
========
[X] make TriggerMode class an Enum
[X] setup minimal connection without changing settings other than buffer lengths
[X] extract window things
[ ] cut obtained data to fit into requested windows
[ ] provide interface for changing trigger settings
[ ] print information about how long the measurement is expected to run
[ ] implement multiple triggers (using rows) (and check how this actually behaves)
	[ ] count
	[ ] endless
	[ ] change in trigger input port
[X] implement setting recording channel (could that be already something inside qupulse?)
=> this should be sufficient for operation
[ ] implement optional operations (averaging, binning, up/down sampling, ...)
[ ] implement scope interface for higher sample rates (if i understood the documentation correctly)
[ ] implement low level interface (subscribe())

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

		self.daq = self.api_session.dataAcquisitionModule()
		self.daq.set('device', device_serial)

		self.daq.set('type', 1)
		self.daq.set('triggernode', '/dev3442/demods/0/sample.AuxIn0')
		self.daq.set('endless', 0)

		self.daq_read_return = {}

		if reset:
			# Create a base configuration: Disable all available outputs, awgs, demods, scopes,...
			zhinst.utils.disable_everything(self.api_session, self.serial)

		self.programs = {}
		self.currently_set_program = None

	
	def reset_device(self):
		""" This function resets the device to a known default configuration.
		"""

		raise NotImplementedError()

		self.clear()

	def register_measurement_channel(self, program_name:str=None, channel_path:Union[str, List[str]]=[], window_name:str=None):
		""" This function saves the channel one wants to record with a certain program

		Args:
			program_name: Name of the program
			channel_path: the channel to record in the shape of "demods/0/sample.R.avg". Note that everything but the things behind the last "/" are considered to relate do the demodulator. If this is not given, you might want to check this driver and extend its functionality.
			window_name: The windows for that channel.

		"""

		if not isinstance(channel_path, list):
			channel_path = [channel_path]
		self.programs.setdefault(program_name, {}).setdefault("channel_mapping", {}).setdefault(window_name, []).extend(channel_path)

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

		self.programs.setdefault(program_name, {}).setdefault("windows", {}).update(windows)

		for k, v in windows.items():
			self.set_measurement_mask(program_name=program_name, mask_name=k, begins=v[0], lengths=v[1])

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

		assert begins.shape == lengths.shape

		if program_name not in self.programs:
			raise ValueError(f"Program '{program_name}' not known")

		# the channels we want to measure with:
		channels_to_measure: List[str] = []
		try:
			channels_to_measure = self.programs[program_name]["channel_mapping"][mask_name]
		except KeyError:
			try:
				channels_to_measure = self.programs[None]["channel_mapping"][mask_name]
			except KeyError:
				channels_to_measure = []

		if len(channels_to_measure) == 0:
			warnings.warn(f"There are no channels defined that should be measured in mask '{mask_name}'.")

		# get the sample rates for the requested channels. If no sample rate is found, None will be used. This code is not very nice.
		currently_set_sample_rates: List[Union[TimeType, None]] = []

		for c in channels_to_measure:
			currently_set_sample_rates.append(self._get_sample_rates(c))


		mask_info = np.full((3, len(begins), len(currently_set_sample_rates)), np.nan)

		for i, sr in enumerate(currently_set_sample_rates):
			if sr is not None:
				# this code was taken from the already implemented alazar driver. 
				mask_info[0, :, i] = np.rint(begins * float(sr)).astype(dtype=np.uint64) # the begin
				mask_info[1, :, i] = np.floor_divide(lengths * float(sr.numerator), float(sr.denominator)).astype(dtype=np.uint64) # the length
				mask_info[2, :, i] = (mask_info[0, :, i] + mask_info[1, :, i]).astype(dtype=np.uint64) # the end

		self.programs.setdefault(program_name, {}).setdefault("masks", {})[mask_name] = {"mask": mask_info, "channels": channels_to_measure, "sample_rates": currently_set_sample_rates}
		self.programs.setdefault(program_name, {}).setdefault("all_channels", set()).update(channels_to_measure)
		self.programs.setdefault(program_name, {}).setdefault("window_hull", [np.nan, np.nan])

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
			else:
				_start = np.nanmin(mask_info[0])
				_end = np.nanmax(mask_info[2])
				if np.isnan(self.programs[program_name]["window_hull"][0]) or self.programs[program_name]["window_hull"][0] > _start:
					self.programs[program_name]["window_hull"][0] = _start
				if np.isnan(self.programs[program_name]["window_hull"][1]) or self.programs[program_name]["window_hull"][1] > _end:
					self.programs[program_name]["window_hull"][1] = _end

			return (np.min(mask_info[0], axis=-1), np.max(mask_info[2], axis=-1))

	def register_operations(self, program_name: str, operations) -> None:
		"""Register operations that are to be applied to the measurement results.

		Args:
			program_name: Name of the program
			operations: DAC specific instructions what to do with the data recorded by the device.
		"""

		self.programs.setdefault(program_name, {}).setdefault("operations", []).append(operations)
	
	def _get_demod(self, channel:str):
		""" This function gets the demodulator corresponding to a channel
		"""
		elements = channel.split("/")
		elements = [e for e in elements if not e.lower()==self.serial.lower()]
		elements = [e for e in elements if len(e) > 0]

		return "/".join(elements[:-1])

	def arm_program(self, program_name: str, force:bool=True) -> None:
		"""Prepare the device for measuring the given program and wait for a trigger event."""

		# check if program_name specified program is selected and important parameter set to the lock-in
		if self.currently_set_program is None or self.currently_set_program != program_name or force:

			for c in self.programs[program_name]["all_channels"]:

				# activate corresponding demodulators
				demod = self._get_demod(c)
				try:
					self.api_session.setInt(f'/{self.serial}/{demod}/enable', 1)
				except RuntimeError  as e:
					if "ZIAPINotFoundException" in e.args[0] or f"Path /{self.serial}/{demod}/enable not found." in e.args[0]:
						# ok, the channel can not be enabled. Then the user should be caring about that.
						warnings.warn(f"The channel {c} does not have an interface for enabling it. If needed, this can be done using the web interface.")
						pass	
					else:
						raise

				# select the value to measure
				self.daq.subscribe(f'/{self.serial}/{c}')

			# check if sample rates are the same as when register_measurement_windows() was called
			for k, v in self.programs[program_name]['masks'].items():
				if len(v["channels"]) != len(v["sample_rates"]):
					raise ValueError(f"There is a mismatch between number the channels to be used and the known sample rates.")
				for c, r in zip(v["channels"], v["sample_rates"]):
					if self._get_sample_rates(c) != r:
						raise ValueError(f"The sample rate for channel '{c}' has changed. Please call register_measurement_windows() again.")

			# set daq module settings to standard things
			# TODO one might want to extend the driver to support more methods
			self.daq.set('grid/mode', 4) # this corresponds to Mode: Exact(on-grid)
			self.daq.set('grid/rows', 1) # this corresponds to measuring only for one trigger
			# the following two lines set the row repetitions to 1 and off
			self.daq.set('grid/repetitions', 1)
			self.daq.set('grid/rowrepetition', 0)
			# TODO these should be a

			# set the buffer size according to the largest measurement window
			# TODO one might be able to implement this a bit more cleverly
			self.daq.set('grid/cols', int(self.programs[program_name]["window_hull"][1]))

			self.currently_set_program = program_name

		# execute daq
		self.daq.execute()

		# wait until changes have taken place
		self.api_session.sync()

		# # TODO this should be redundant with self.currently_set_program
		# if program_name != None:
		# 	self.programs.setdefault(program_name, {}).setdefault('armed', False)
		# 	self.programs[program_name]['armed'] = True

	def unarm_program(self, program_name:str):
		""" unarms the lock-in. This should be program independent.
		"""

		self.daq.finish()
		self.daq.unsubscribe('*')
		self.api_session.sync()

		self.currently_set_program = None
		
		# # TODO this should be redundant with self.currently_set_program
		# if program_name != None:
		# 	self.programs.setdefault(program_name, {}).setdefault('armed', False)
		# 	self.programs[program_name]['armed'] = False

	def force_trigger(self, program_name:str):
		""" forces a trigger
		"""

		self.daq.set('forcetrigger', 1)

	def delete_program(self, program_name: str) -> None:
		"""Delete program from internal memory."""

		# this does not have an effect on the current implementation of the lock-in driver.

		if self.currently_set_program == program_name:
			self.unarm_program(program_name)

		self.programs.pop(program_name)

	def clear(self) -> None:
		"""Clears all registered programs."""

		self.unarm_program(program_name=None)

	def measure_program(self, channels: Iterable[str], wait=True) -> Dict[str, np.ndarray]:
		"""Get the last measurement's results of the specified operations/channels"""

		# wait until the data acquisition has finished
		while not self.daq.finished() and wait:
			time.sleep(1)
			print(f"waiting for device {self.serial} to finish the acquisition.")

		if not self.daq.finished():
			raise ValueError(f"Device {self.serial} did not finish the acquisition in time.")

		data = self.daq.read()
		self.daq_read_return.update(data)

		# go through the returned object and extract the data of interest

		recorded_data = {}

		for device_name, device_data in data.items():
			if device_name == self.serial:
				for input_name, input_data in device_data.items():
					for signal_name, signal_data in input_data.items():
						for final_level_name, final_level_data in signal_data.items():
							channel_name = f"/{device_name}/{input_name}/{signal_name}/{final_level_name}"
							channel_data = [xr.DataArray(
										data=d["value"],
										coords={'timestamp': (['col', 'row'], d['timestamp'])},
										dims=['col', 'row'],
										name=channel_name,
										attrs=d['header']) 
								for i, d in enumerate(final_level_data)]
							recorded_data[channel_name] = channel_data


		return recorded_data


		# the first dimension of channel_data is expected to be the history of multiple not read data points. This will be handled as multiple entries in a list. This will then not make too much sense, if not every channel as this many entries. If this is the case, they will be stacked, such that for the last elements it fits.
		# TODO do this based on the timestamps and not the indices. That might be more sound than just assuming that.


		# applying measurement windows and optional operations
		# TODO implement operations

		# targeted structure:
		# results[<mask_name>][<channel>] -> [data]

		masked_data = {}

		for mask_name in self.programs.[self.currently_set_program]["masks"]:
			pass

		result = xr.Dataset(
			{}
			)
		# self.programs.setdefault(program_name, {}).setdefault("masks", {})[mask_name] = {"mask": mask_info, "channels": channels_to_measure, "sample_rates": currently_set_sample_rates}

		
		print(data)
