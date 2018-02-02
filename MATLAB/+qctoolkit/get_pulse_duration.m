function pulse_length = get_pulse_duration(pulse, parameters)
	
parameter_kwargs = cell2namevalpairs(fieldnames(parameters), struct2cell(parameters));
pulse_length = pulse.duration.evaluate_numeric(pyargs(parameter_kwargs{:}))*1e-9;




% delete..
function cellarr = cell2namevalpairs(fieldnames, values)
	cellarr={};
	for ii = 1:numel(fieldnames)
		cellarr{end+1}=fieldnames{ii};
		cellarr{end+1}=values{ii};
	end

	