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

GPADAT_ADDR = 0x006FC0
SCRIPT_START_ADDRESS = 49408


def volt_to_int(voltage, offset:float=-10.0, rng:float=20):
	return max(min((2**16-1), int((voltage-offset)/rng * (2**16-1))), 0)

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

	resp = serial_ask("X{};"%(label))

	return resp

def stop_script(serial_ask):

	return run_script(serial_ask, label=0)



