function pulse = load_pulse(pulse_name)
	
	global plsdata
    % todo: add a method to sync with filesystem
    plsdata.qc.pulse_storage.clear()
	pulse = plsdata.qc.pulse_storage{pulse_name};