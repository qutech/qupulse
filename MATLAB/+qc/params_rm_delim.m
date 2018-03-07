function [parameters, pulse_name] = params_rm_delim(parameters)
	% Remove prefix from parameters identified by three underscores
  % (needed for dictionaries).

	delim = '___';
	fn = fieldnames(parameters)';
	[i1, i2] = regexp(fn{1}, '^.+___');
	pulse_name = fn{1}(i1:i2-numel(delim));

	for fn = fieldnames(parameters)'		
		parameters.(fn{1}(i2+1:end)) = parameters.(fn{1});
		parameters = rmfield(parameters, fn{1});
	end