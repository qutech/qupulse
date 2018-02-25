function parameters = join_params_and_dicts(varargin)
	% Each argument is either a dictionary name as a string, a dictionary
	% struct or a parameter struct. This function joins all input arguments to
	% a single parameters struct. If conflicts occur, the field of the rightmost
	% argument takes precedence.
	%
	% Dictionaries are saved as json files in the repository qctoolkit-dicts.
	% Each dictionary is a struct with a field for each pulse. The field names
	% are the same as the pulse names (as pulse names are unique identifiers).
	% Each field has subfields which store pulse parameters for that pulse.
	%
	% When defining pulses in qctoolkit, prefix each parameter name with the
	% pulsename followed by three underscores. I.e. for a pules 'meas' the
	% parameter 'waiting_time' should be called 'meas___waiting_time'. The
	% dictionary however should only have the parameter field 'waiting_time',
	% i.e struct('meas', struct('waiting_time', 1)).
	%
	% Each dictionary also has a global field which also contains a struct. The
	% field names of this struct refer to parameters which can be set to the
	% same value across different pulses in the same dictionary. For example
	% struct('global', struct('waiting_time', 1)) will set the parameter
	% 'waiting_time' to 1 for all pulses which have the parameter
	% 'waiting_time' in the dictionary where the global is defined.
	%
	% Globals do not take precedence over parameters passed in more to the
	% right.
	%
	% Each dictionary also has a field 'dict___name' which specifies
  % the dictionary name

	p = varargin;
	p = cellfun(@load_dict, p, 'UniformOutput', false);
	p = cellfun(@apply_globals, p, 'UniformOutput', false);
	p = cellfun(@dict_to_parameter_struct, p, 'UniformOutput', false);

	parameters = struct();
	for k = 1:numel(p)
		parameters = join(parameters, p{k});
	end

end


function d = load_dict(d)
	if is_dict(d) && ischar(d)
		d = qc.load_dict(d);
	end	
end

function bool = is_dict(dp)
	delim = '___';	
	bool = ischar(dp) || (isstruct(dp) && isfield(dp, strcat('dict', delim, 'name')));
end

function d = apply_globals(d)
	% Replace all parameters by their global values	
	if is_dict(d) && isfield(d, 'global')
		delim = '___';
		globals = fieldnames(d.global);
		
		for pulseName = fieldnames(d)'
			if strcmp(pulseName{1}, 'global') || strcmp(pulseName{1}, strcat('dict', delim, 'name'))
				continue
			end
			for paramName = fieldnames(d.(pulseName{1}))'
				
				bool = cellfun(@(x)(strcmp(paramName{1}, x)), globals, 'UniformOutput', true);
				
				if any(bool)
					d.(pulseName{1}).(paramName{1}) = d.global.(globals{bool});
				end
			end
		end
		d = rmfield(d, 'global');
	end	
end

function p = dict_to_parameter_struct(d)
	% Flatten dict into a parameter struct
	if is_dict(d)
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

function p = join(p1, p2)
	% Join two parameter structs. This process is additive, fields in p2 take
	% precedence
	p = p1;

	for paramName = fieldnames(p2)'
		p.(paramName{1}) = p2.(paramName{1});
	end	
end



