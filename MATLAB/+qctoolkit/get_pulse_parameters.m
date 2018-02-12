function pulse_parameters = get_pulse_parameters(varargin)
	
	pulse_template = qctoolkit.load_pulse(varargin{:});
	
	pulse_parameters = util.py.py2mat(pulse_template.parameter_names);
	