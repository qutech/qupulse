function pulseParameters = get_pulse_params(pulse_name_or_template)
	
	if ischar(pulse_name_or_template)
		pulse_template = qc.load_pulse(pulse_name_or_template);
	else
		pulse_template = pulse_name_or_template;
	end
	pulseParameters = sort(util.py.py2mat(pulse_template.parameter_names));
	