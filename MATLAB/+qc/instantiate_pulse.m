function instantiated_pulse = instantiate_pulse(pulse, varargin)
	% Plug in parameters
	
	if qc.is_instantiated_pulse(pulse)
		instantiated_pulse = pulse;
		
	else
		default_args = struct(...
			'parameters', py.None, ...
			'channel_mapping', py.None, ...
			'window_mapping' , py.None ...
			);
		
		args = util.parse_varargin(varargin, default_args);
		
		sequencer = py.qctoolkit.pulses.Sequencer();
		
		args.channel_mapping = replace_empty_with_pynone(args.channel_mapping);
		args.window_mapping = replace_empty_with_pynone(args.window_mapping);
		
		kwargs = pyargs( ...
			'parameters' ,     args.parameters, ...
			'channel_mapping', args.channel_mapping, ...
			'window_mapping' , args.window_mapping ...
			);
				
		sequencer.push(pulse, kwargs);	
		instantiated_pulse = util.py.call_with_interrupt_check(py.getattr(sequencer, 'build'));
	end
end


function mappingStruct = replace_empty_with_pynone(mappingStruct)
	
	for fn = fieldnames(mappingStruct)'
		if isempty(mappingStruct.(fn{1}))
			mappingStruct.(fn{1}) = py.None;
		end
	end
	
end

