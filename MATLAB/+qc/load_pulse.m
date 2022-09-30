function pulse = load_pulse(pulse_name)
	
	global plsdata
	pulse = plsdata.qc.pulse_storage{pulse_name};