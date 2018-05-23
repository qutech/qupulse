function d = dict_apply_globals(d)
	% Replace all parameters by their global values	
	if qc.is_dict(d) && isfield(d, 'global')
		delim = '___';
		globals = fieldnames(d.global);
		
		for pulseName = fieldnames(d)'
			if strcmp(pulseName{1}, strcat('dict', delim, 'name'))
				continue
			end
			for paramName = fieldnames(d.(pulseName{1}))'
				
				bool = cellfun(@(x)(strcmp(paramName{1}, x)), globals, 'UniformOutput', true);
				
				if any(bool)
					d.(pulseName{1}).(paramName{1}) = d.global.(globals{bool});
				end
			end
		end
	end	
end