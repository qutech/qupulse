function pulse_length = get_pulse_duration(pulse_template, parameters)
	% Return pulse length in s
		
	kwargs = cell2namevalpairs(fieldnames(parameters), struct2cell(parameters));
	pulse_length = py.getattr(pulse_template, 'duration').evaluate_numeric(pyargs(kwargs{:}))*1e-9;
	
		
function cellarr = cell2namevalpairs(field_names, values)
	cellarr={};
	for k = 1:numel(field_names)
		cellarr{end+1}=field_names{k};
		cellarr{end+1}=values{k};
	end
	
	