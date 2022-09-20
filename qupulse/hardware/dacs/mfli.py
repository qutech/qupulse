""" This file contains a qupulse driver for the DAQ module of an Zuerich Instruments MFLI.

May lines of code have been adapted from the zihdawg driver.

"""
from typing import Dict, Tuple, Iterable
from enum import Enum
import warnings

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
[ ] setup minimal connection without changing settings other than buffer lengths
[ ] extract window things
[ ] implement multiple triggers (using rows) (and check how this actually behaves)
	[ ] count
	[ ] endless
	[ ] change in trigger input port
[ ] make sample rate clean
[ ] implement setting recording channel (could that be already something inside qupulse?)
=> this should be sufficient for operation
[ ] have an interface for setting trigger configurations
[ ] an interface for setting the sample rate based on more convenient parameters
=> this should cover the common use cases
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
		self.device = self.api_session.connectDevice(device_serial, device_interface)
		self.default_timeout = timeout
		self.serial = device_serial

		self.daq = self.api_session.dataAcquisitionModule()

		if reset:
			# Create a base configuration: Disable all available outputs, awgs, demods, scopes,...
			zhinst.utils.disable_everything(self.api_session, self.serial)

		self.programs = {}

	
	def reset_device(self):
		""" This function resets the device to a known default configuration.
		"""

		raise NotImplementedError()

		self.clear()

	def register_measurement_channel(self, program_name:str, channel_path:str):
		""" This function saves the channel one wants to record with a certain program

		Args:
			program_name: Name of the program
			channel_path: the channel to record in the shape of "demods/0/sample.R.avg". Note that everything but the things behind the last "/" are considered to relate do the demodulator. If this is not given, you might want to check this driver and extend its functionality.

		"""

		if not isinstance(channel_path, list):
			channel_path = [channel_path]

		self.programs.setdefault(program_name, {}).setdefault("channels", [])
		self.programs[program_name]["channels"] = channel_path

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

		# get the sample rates for the requested channels. If no sample rate is found, None will be used. This code is not very nice.
		currently_set_sample_rates: List[Union[TimeType, None]] = []
		for c in self.programs[program_name]["channels"]:
			try:
				timetype_sr = TimeType().from_float(value=self.api_session.getDouble(f"/{self.serial}/{self._get_demod(c)}/rate"), absolute_error=0)
				currently_set_sample_rates.append(timetype_sr)
			except RuntimeError  as e:
				if "ZIAPINotFoundException" in e.args[0]:
					currently_set_sample_rates.append(None)
				else:
					raise


		mask_info = np.full((3, len(begins), len(currently_set_sample_rates)), np.nan)

		for i, sr in enumerate(currently_set_sample_rates):
			if sr is not None:
				# this code was taken from the already implemented alazar driver. 
				mask_info[0, :, i] = np.rint(begins * float(sr)).astype(dtype=np.uint64) # the begin
				mask_info[1, :, i] = np.floor_divide(lengths * float(sr.numerator), float(sr.denominator)).astype(dtype=np.uint64) # the length
				mask_info[2, :, i] = (mask_info[0, :, i] + mask_info[1, :, i]).astype(dtype=np.uint64) # the end

		self.programs.setdefault(program_name, {}).setdefault("masks", {})[mask_name] = mask_info

		# as the lock-in can measure multiple channels with different sample rates, the return value of this function is not defined correctly.
		# this could be fixed by only measuring on one channel, or by returning some "summary" value. As of now, there does not to be a use of the return values.

		return (np.min(mask_info[0], axis=-1), np.min(mask_info[2], axis=-1))

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

	def arm_program(self, program_name: str) -> None:
		"""Prepare the device for measuring the given program and wait for a trigger event."""

		# TODO check if program_name specified program is selected and important parameter set to the lock-in

		for c in self.programs[program_name].channels:
			# select the value to measure
			self.daq.subscribe(f'/{self.serial}/{c}')

			# activate corresponding demodulators
			demod = self._get_demod(c)
			try:
				self.daq.set(f'/{self.serial}/{demod}/enable', 1)
			except RuntimeError  as e:
				if "ZIAPINotFoundException" in e.args[0]:
					# ok, the channel can not be enabled. Then the user should be caring about that.
					warnigns.warn(f"The channel {c} does not have an interface for enabling it. If needed, this can be done using the web interface.")
					pass	
				else:
					raise

		# execute daq
		self.daq.execute()

		# wait until changes have taken place
		self.daq.sync()

		if program_name != None:
			self.programs.setdefault(program_name, {}).setdefault('armed', False)
			self.programs[program_name]['armed'] = True

		raise NotImplementedError()

	def unarm_program(self, program_name:str):
		""" unarms the lock-in. This should be program independent.
		"""

		self.daq.finish()
		self.daq.unsubscribe('*')
		self.daq.sync()
		
		if program_name != None:
			self.programs.setdefault(program_name, {}).setdefault('armed', False)
			self.programs[program_name]['armed'] = False

	def force_trigger(self, program_name:str):
		""" forces a trigger
		"""

		self.daq.set('forcetrigger', 1)

	def delete_program(self, program_name: str) -> None:
		"""Delete program from internal memory."""

		# this does not have an effect on the current implementation of the lock-in driver.

		pass

	def clear(self) -> None:
		"""Clears all registered programs."""

		self.unarm_program(program_name=None)

	def measure_program(self, channels: Iterable[str]) -> Dict[str, np.ndarray]:
		"""Get the last measurement's results of the specified operations/channels"""

		data = self.daq.read()
		
		print(data)