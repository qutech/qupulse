function reduced_program = get_minimal_program(program)
	% Return program with all parameters not needed by its pulse template
	% removed
	
	pulse_template = qc.struct_to_pulse(program.pulse_template);
	parameter_names = util.py.py2mat(pulse_template.parameter_names);
	
	program.parameters_and_dicts = qc.join_params_and_dicts(program.parameters_and_dicts);
	reduced_program = program;	
	reduced_program.parameters_and_dicts = struct();
	
	% Remove all fields in parameters_and_dicts not needed by pulse_template
	for p = parameters_names'		
		reduced_program.parameters_and_dicts.(p{1}) = program.parameters_and_dicts.(p{1});		
	end
	
	
	