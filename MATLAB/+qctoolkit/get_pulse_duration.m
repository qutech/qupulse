function pulse_length = get_pulse_duration(pulse, parameters)
	
parameter_kwargs = util.zip(fieldnames(parameters), struct2cell(parameters));
pulse_length = pulse.duration.evaluate_numeric(pyargs(parameter_kwargs{:}))*1e-9;