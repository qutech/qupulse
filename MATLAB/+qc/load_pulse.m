function pulse = load_pulse(pulse_name)
	
	global plsdata
	pulse = plsdata.qc.serializer.deserialize(pulse_name);