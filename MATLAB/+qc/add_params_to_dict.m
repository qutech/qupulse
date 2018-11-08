function d = add_params_to_dict(d, parameters, pulse_name)
	
	if nargin < 3
		pulse_name= [];
	end

	delim = '___';
	
	fn = fieldnames(parameters)';

	if ~isempty(fn) && util.str_contains(fn{1}, delim)
		[parameters, extracted_pulse_name] = qc.params_rm_delim(parameters);

		if isempty(pulse_name)
			pulse_name = extracted_pulse_name;
		end
	end

	if isempty(pulse_name)
		error('Pulse name must not be empty');
	end

	d = qc.load_dict(d);
	d.(pulse_name) = parameters;
	
	
