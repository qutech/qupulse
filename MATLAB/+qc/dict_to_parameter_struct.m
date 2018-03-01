function p = dict_to_parameter_struct(d)
	% Flatten dict into a parameter struct
	if qc.is_dict(d)
		delim = '___';
		d = rmfield(d, strcat('dict', delim, 'name'));
		
		p = {};
		for pulseName = fieldnames(d)'
			for paramName = fieldnames(d.(pulseName{1}))'
				p{end+1} = strcat(pulseName{1}, delim, paramName{1});
				p{end+1} = d.(pulseName{1}).(paramName{1});
			end
		end
		p = struct(p{:});
	else
		p = d;
	end
end