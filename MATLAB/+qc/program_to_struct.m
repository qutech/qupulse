function program = program_to_struct(program_name, pulse_template, parameters_and_dicts, channel_mapping, window_mapping, global_transformation)
		
	if ischar(pulse_template)
		error('Variable ''pulse_template'' must not be of type char to make sure the correct pulse is saved. Please pass a pulse template.')
	end
	
	% Make sure all dictionaries are loaded so not just saving strings
	if ~iscell(parameters_and_dicts)
		parameters_and_dicts = {parameters_and_dicts};
	end	
	parameters_and_dicts = cellfun(@qc.load_dict, parameters_and_dicts, 'UniformOutput', false);
	
	program = struct( ...
		'program_name',          program_name, ...
		'pulse_template',        qc.pulse_to_struct(pulse_template), ...
		'parameters_and_dicts',  {parameters_and_dicts}, ...
		'channel_mapping',       channel_mapping, ...
		'global_transformation', global_transformation, ...
		'window_mapping',        window_mapping ...
		);			
	program.pulse_duration = qc.get_pulse_duration(pulse_template, qc.join_params_and_dicts(program.parameters_and_dicts));
	program.added_to_pulse_duration	= 0;
	
	for name = fieldnames(program.channel_mapping)'
		if strcmp(class(program.channel_mapping.(name{1})), 'py.NoneType')
			program.channel_mapping.(name{1}) = py.None;
		end
	end
	
	for name = fieldnames(program.window_mapping)'
		if strcmp(class(program.window_mapping.(name{1})), 'py.NoneType')
			program.window_mapping.(name{1}) = py.None;
		end
	end
	
	