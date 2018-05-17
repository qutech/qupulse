function dict_string_or_struct = load_dict(dict_string_or_struct)
	% Load dict if d is a string. Otherwise leave d untouched.
	% Important: this does not (re)load a dict if the passed in variable is
	% already a struct
	global plsdata
	delim = '___';
	
	if ischar(dict_string_or_struct)
		file_name = fullfile(plsdata.dict.path, [dict_string_or_struct '.json']);
		if exist(file_name, 'file')
			text = fileread(file_name);
			dict_string_or_struct = jsondecode(text);
			dict_string_or_struct = qc.array2row(dict_string_or_struct);
		else
			dict_string_or_struct = struct(strcat('dict', delim, 'name'), dict_string_or_struct, 'global', struct());
		end
	end
	
end