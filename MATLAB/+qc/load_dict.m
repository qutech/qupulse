function dict_string_or_struct = load_dict(dict_string_or_struct)
	% Load dict if d is a string. Otherwise leave d untouched.
	%
	% Important: this does not (re)load a dict if the passed in variable is
	% already a struct.
	%
	% You can specifiy a suffix to be appended to all pulse names in the
	% dictionary separated by a space, i.e. if the dictionary name is
	% 'common' you can pass 'common d12' and '_d12' will be appended to each
	% pulse name.
	
	global plsdata
	delim = '___';
	
	if ischar(dict_string_or_struct)
		dict_string_or_struct = strsplit(dict_string_or_struct, ' ');	
		
		if numel(dict_string_or_struct) > 1
			suffix = ['_' dict_string_or_struct{2}];
		else
			suffix = '';
		end
		dict_string_or_struct = dict_string_or_struct{1};
		
		file_name = fullfile(plsdata.dict.path, [dict_string_or_struct '.json']);
		if exist(file_name, 'file')
			text = fileread(file_name);
			dict_string_or_struct = jsondecode(text);
			dict_string_or_struct = qc.array2row(dict_string_or_struct);
		else
			dict_string_or_struct = struct(strcat('dict', delim, 'name'), dict_string_or_struct, 'global', struct());
		end
		
		if ~strcmp(suffix, '')
			for fn = fieldnames(dict_string_or_struct)'		
				if ~strcmp(fn{1}, 'global') && ~strcmp(fn{1}, strcat('dict', delim, 'name'))
					dict_string_or_struct.([fn{1} suffix]) = dict_string_or_struct.(fn{1});
					dict_string_or_struct = rmfield(dict_string_or_struct, fn{1});
				end
			end
		end
		
	end
	
end