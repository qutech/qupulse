function instantiated_pulse = instantiate_pulse(pulse, varargin)
	% Plug in parameters
	
	if qc.is_instantiated_pulse(pulse)
		instantiated_pulse = pulse;
		
	else
		default_args = struct(...
			'parameters', py.None, ...
			'channel_mapping', py.None, ...
			'window_mapping' , py.None, ...
			'global_transformation', [], ...
			'to_single_waveform', py.set() ...
			);
		
		args = util.parse_varargin(varargin, default_args);
		
		args.channel_mapping = replace_empty_with_pynone(args.channel_mapping);
		args.window_mapping = replace_empty_with_pynone(args.window_mapping);
		args.global_transformation = qc.to_transformation(args.global_transformation);
		
		kwargs = pyargs( ...
			'parameters' ,     args.parameters, ...
			'channel_mapping', args.channel_mapping, ...
			'measurement_mapping' , args.window_mapping, ...
			'global_transformation', args.global_transformation, ...
			'to_single_waveform', args.to_single_waveform ...
			);
		
		instantiated_pulse = util.py.call_with_interrupt_check(py.getattr(pulse, 'create_program'), kwargs);
	end
end


function mappingStruct = replace_empty_with_pynone(mappingStruct)
	
	for fn = fieldnames(mappingStruct)'
		if isempty(mappingStruct.(fn{1}))
			mappingStruct.(fn{1}) = py.None;
		end
	end
	
end

