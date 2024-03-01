""" this module contains helper functions to work with the DecaDAC
"""

from typing import *
from types import MappingProxyType
import numpy as np
import contextlib
import itertools
import functools
import warnings
import logging
import tqdm

from qupulse.program.linspace import Command, LoopLabel, Increment, Set as _Set, Wait, LoopJmp, Play, reduce_commands
from qupulse.utils.performance import *

GPADAT_ADDR = 0x006FC0
SCRIPT_START_ADDRESS = 49408


def volt_to_int(voltage, offset:float=-10.0, rng:float=20):
	return np.maximum(np.minimum((2**16-1), np.floor((voltage-offset)/rng * (2**16-1))), 0)

def default_ask(serial, cmd):
	if not isinstance(cmd, bytes):
		cmd = cmd.encode('ascii')
	serial.write(cmd)
	time.sleep(0.05)
	serial.flush()
	resp = serial.read_all()
	return resp

def inject_int(target_address:int, value:int, serial_ask:Callable):
	resp = serial_ask('A%i;P%i;'%(target_address, value))
	return resp


def inject_ascii(position:int, string:str, serial_ask:Callable, fix_1:bool=True, script_offset:int=SCRIPT_START_ADDRESS):
	""" This method overwrites 2 chars of the uploaded script
	"""
	assert position%2==0
	assert len(string) == 2, "Only 2 chars (2*8bit = 16bit) are supported."
	
	string = string[::-1]
	
	target_address = np.ceil(position/2).astype(int)+script_offset
	value = int(bin(int.from_bytes(string.encode(), 'big')), 2)

	resp = inject_int(target_address, value)
	
	if fix_1:
		""" 
		The Problem: the overwritten chars are also written to the first 16 byte.
		When running the inject call to overwrite it back to "*1" things work,.
		But not in this if here. also not with the sleep... (but when prun timed, the call is below 1ms (but probably close to 1ms))
		But reading requires a wait of ~60ms...
		"""
		
		time.sleep(0.5)
		inject_ascii(position=0, string="*1", fix_1=False, script_offset=script_offset)

def download_16bit(index, serial_ask) -> int:
	""" This function download 16bit from an arbitrary position within the whole memory.
	"""
	resp = serial_ask('A%i;p;&p;'%(index))
	
	s = str(resp).split("!")
	decimal_rep = int(s[1][1:])

	return decimal_rep


def get_chars_from_script(index:int, serial_ask, script_offset:int=SCRIPT_START_ADDRESS) -> List[str]:
	""" This function gets a char at a given position from the uploaded scan. This function reads a 16 bit block, which contains 2 8bit chars.
	"""

	decimal_rep = download_16bit(index+script_offset, serial_ask=serial_ask)
	string = f"{bin(decimal_rep)[2:]:0>16}"
	
	res = []
	for sub in [string[8:], string[:8]]:
		try: 
			c1 = int(f"0b{sub}", 2)
			c1 = c1.to_bytes((c1.bit_length() + 7) // 8, 'big').decode()
			res.append(c1)
		except:
			print(f"Could not decode {sub}")
			pass
	
	return res

def download_script(script_length:int, serial_ask, script_offset=SCRIPT_START_ADDRESS) -> str:
	""" This function downloads the uploaded script.
	"""

	res = []
	for i in tqdm.tqdm(range(np.ceil(script_length/2).astype(int))):
		res.extend(get_chars_from_script(i, serial_ask=serial_ask, script_offset=script_offset))
	return "".join(res)


def upload_script(script:str, serial_ask) -> str:
	""" This function uploads a given script.
	"""

	# removes the comments behind each line:
	script = "\n".join([l.split("(")[0] for l in script.split("\n")])
	
	script = script.replace(' ', '')
	script = script.replace('\n', '')
	assert script.startswith("{")
	assert script.endswith("};")

	resp = serial_ask(script)

	return resp

def run_script(serial_ask, label:int=1):

	resp = serial_ask(f"X{label};")

	return resp

def stop_script(serial_ask):

	return run_script(serial_ask, label=0)

def LZ77_to_linspace_commands(lz77_compressed, dt=1e+5) -> List[Command]:
	""" This function creates a list of Linspace Commands from a lz77 compressed pulse. The LZ77 compression requires allow_intermediates=False, allow_reconstructions_using_reconstructions=False. Further using_diffs=True should be helpful.
	"""

	print_intermediate_programs = False
	print_index_calculations = False

	if isinstance(dt, (int, float)):
		dt = np.ones(len(lz77_compressed))*dt
	assert len(dt) == len(lz77_compressed)

	loop_indecies = range(int(10_000_000_000_000)).__iter__()

	commands = []
	unrolled_step_index = []
	for s, (o, d, v) in enumerate(lz77_compressed):
		if print_intermediate_programs or print_index_calculations: print(); 
		if print_intermediate_programs: print((o, d, v))
		commands.append([])
		unrolled_step_index.append((unrolled_step_index[-1]+1 if len(unrolled_step_index)>0 else 0))
		if v is not None:
			for i, a in enumerate(v):
				if a != 0:
					commands[-1].append(Increment(i, a, None))
			commands[-1].append(Wait(dt[s]))
		if d != 0:
			lx = next(loop_indecies)
			r = d//o
			assert r > 0
			if print_index_calculations: 
				print(o, len(commands))
				print((o, d, v))
				print(s, (o, d), r, unrolled_step_index, unrolled_step_index[-1]-o)
			temp = np.where(np.array(unrolled_step_index) == unrolled_step_index[-1]-o)[0]
			assert len(temp) > 0, "Something seams to be off with the indexing. Likely due to jumping somewhere into a loop and not to its start."
			eff = temp[-1]
			if print_index_calculations: print(eff)
			commands[eff].insert(0, LoopLabel(lx, r+1))
			commands[-2].append(LoopJmp(lx))
			foo = unrolled_step_index[-1]
			for _o in range(len(unrolled_step_index)):
				if len(unrolled_step_index)-_o-1-1 < eff: break
				if print_index_calculations: print(_o, r, _o)
				unrolled_step_index[-_o-1] += d
				if print_index_calculations: print(unrolled_step_index)
		if print_index_calculations: print(unrolled_step_index)
		if print_intermediate_programs: print("\n".join([str(c) for c in commands]))

	if print_intermediate_programs or print_index_calculations: print(commands)

	# flattening the commands
	commands = [c for cc in commands for c in cc]

	return commands

def translate_command_list_to_ascii(commands:List[Command], channel_mapping:Union[Dict[int, int], None]=None, volt_in_dac_basis:bool=False) -> str:

	boards:Union[List[int], None] = None # TODO get this list from the used channel
	
	if channel_mapping is None:
		channel_mapping = {i:i for i in range(5*4)}

	loops = {}
	last_used_loop_addrs = 1
	last_used_memory_addrs = 45056

	res = ["*1:"]

	if boards is None:
		boards = [0, 1, 2, 3, 4]
	for b in boards:
		res.append(f"B{b};M2;")

	if volt_in_dac_basis:
		conversion_fn = lambda v: v
	else:
		conversion_fn = volt_to_int
	   

	# setting up the main trigger
	"""
	SpecialConditions:
	;08 = Trig1 is zero
	;09 = Trig1 is not zero
	;0A = Trig2 is zero
	;0B = Trig2 is not zero
	"""
	last_used_loop_addrs += 1
	res.append(f"*{last_used_loop_addrs}:X{0x800+last_used_loop_addrs};")

	for cmd in commands:
		if cmd.__class__.__name__ == "LoopLabel":
			assert cmd.idx not in loops
			last_used_loop_addrs += 1
			last_used_memory_addrs += 1
			loops[cmd.idx] = {
				"count": cmd.count, 
				"loop_addr": last_used_loop_addrs, 
				"counter_addr": last_used_memory_addrs
				}
			res.extend([
				f"A{loops[cmd.idx]['counter_addr']};", 
				f"P{loops[cmd.idx]['count']};", 
				f"*{loops[cmd.idx]['loop_addr']}:"
				])
		elif cmd.__class__.__name__ == "LoopJmp":
			assert cmd.idx in loops
			res.extend([
				f"A{loops[cmd.idx]['counter_addr']};", 
				f"+-1;", 
				f"X{0x500+loops[cmd.idx]['loop_addr']};"
				])
		elif cmd.__class__.__name__ == "Increment":
			if volt_in_dac_basis:
				int_delta = int(conversion_fn(cmd.value))
			else:
				int_delta = int(conversion_fn(cmd.value)-conversion_fn(0))
			res.extend([
				f"A{1545+16*(channel_mapping[cmd.channel])};", 
				f"+{int_delta};", 
				f"A{1538+16*(channel_mapping[cmd.channel])};", 
				f"P3;"
				])
		elif cmd.__class__.__name__ == "Set":
			value_to_set = int(conversion_fn(cmd.value))
			res.extend([
				f"A{1545+16*(channel_mapping[cmd.channel])};", 
				f"P{value_to_set};", 
				f"A{1538+16*(channel_mapping[cmd.channel])};", 
				f"P3;"
				])
		elif cmd.__class__.__name__ == "Wait":
			d_in_us = int(round(float(cmd.duration)*1e-3)) # from ns to us
			last_used_loop_addrs += 1
			res.extend([
				f"${d_in_us};", 
				f"*{last_used_loop_addrs}:", 
				f"X{0x300+last_used_loop_addrs};"
				])
		else:
			raise NotImplementedError(f"Translating {cmd} is not implemented yet.")

	res.append("X0;")
	
	whole_script = "".join(res)

	assert last_used_loop_addrs < 255, "The program uses too many jump end points."
	assert len(whole_script) < 7680, "The complete program is too long."

	return whole_script

def generate_linspace_commands_from_nparray_using_LZ77(array:np.ndarray, dt:float):
	""" This function creates a list of Commands that represents the given array. This function is build for the DecaDAC.
	"""

	assert len(array.shape) == 2
	assert array.shape[1] <= 20, "The last dimension should be the channel dimension. And our DecaDACs have only 20 channel."

	if isinstance(dt, (float, int)):
		dt = np.ones(array.shape[0])*dt
	assert len(dt.shape) == 1
	assert dt.shape[0] == array.shape[0]
	dt = dt.astype(int)

	# translate the values into DecaDAC values
	array = volt_to_int(array).astype(int)

	# add a time axis
	timed_array = np.concatenate([dt[:, None], array], axis=1)
	timed_array = timed_array.astype(int)
	assert len(timed_array.shape) == 2
	assert timed_array.shape == (array.shape[0], array.shape[1]+1)

	# split off the first sets:
	first_sets = timed_array[0, 1:]
	diff_sets = np.diff(timed_array[:, :], axis=0)
	diff_sets[:, 0] = dt[1:]

	# calculate the LZ77 compression
	comp = compress_array_LZ77(array=diff_sets, allow_intermediates=False, using_diffs=False, allow_reconstructions_using_reconstructions=False)

	# splitting of the time axis
	complz77 = [(o, d, (v[1:] if v is not None else None)) for o, d, v in comp]
	compdt = [(v[0] if v is not None else 0) for o, d, v in comp]

	# generate the list of commands
	comm = LZ77_to_linspace_commands(complz77, dt=compdt)

	# adding the first Set commands and the first wait to the top of the command list
	to_append = [
		_Set(channel=i, value=v, key=None)
		for i, v in enumerate(first_sets)
	] + [Wait(dt[0])]
	comm = to_append + comm

	# reduce unnecessary commands
	comm = reduce_commands(comm)
	return comm

class DecaDACRepresentation:

	def __init__(self, serial_ask:Callable):
		self.serial_ask = serial_ask

	def upload_command_list(self, commands:List[Command], channel_mapping:Union[Dict[int, int], None]=None, volt_in_dac_basis:bool=False):
		ascii_script = translate_command_list_to_ascii(commands, channel_mapping=channel_mapping, volt_in_dac_basis=volt_in_dac_basis)
		print(ascii_script)
		resp = upload_script(script=f"{{{ascii_script}}};", serial_ask=self.serial_ask)

	def upload_numpy_pulse(self, numpy_pulse:np.ndarray, dt:float, channel_mapping:Union[Dict[int, int], None]=None):

		comm = generate_linspace_commands_from_nparray_using_LZ77(numpy_pulse, dt=dt)

		self.upload_command_list(commands=comm, channel_mapping=channel_mapping, volt_in_dac_basis=True)


	def arm(self):
		run_script(serial_ask=self.serial_ask, label=1)

	def reset(self):
		stop_script(serial_ask=self.serial_ask)



