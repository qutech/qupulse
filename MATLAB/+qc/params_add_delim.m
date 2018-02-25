function parameters = params_add_delim(parameters, pulse_name)
	% Prefix all parameters in pulseTemplate with the pulse name followed by
	% three underscores (needed for dictionaries).

	delim = '___';
	for fn = fieldnames(parameters)'		
		parameters.(strcat(pulse_name, delim, fn{1})) = parameters.(fn{1});
		parameters = rmfield(parameters, fn{1});
	end