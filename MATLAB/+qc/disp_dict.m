function text = disp_dict(dict_string_or_struct)
	global plsdata
	delim = '___';

	text = '';	
	
	if isstruct(dict_string_or_struct)
		if ~isfield(dict_string_or_struct, ['dict' delim 'name'])
			error('Please pass a valid dictionary struct. The passed argument is missing the field ''%s''.\n', ['dict' delim 'name']);
		end
		text = py.json.dumps(dict_string_or_struct, pyargs('indent', int8(4), 'sort_keys', true));
		text = char(text);
		dict_string_or_struct = dict_string_or_struct.(['dict' delim 'name']);
	elseif ischar(dict_string_or_struct)
		file_name = fullfile(plsdata.dict.path, [dict_string_or_struct '.json']);
		if exist(file_name, 'file')
			text = fileread(file_name);
		else
			error('Dictionary ''%s'' could not be loaded since file ''%s'' does not exist\n', dict_string_or_struct, file_name);
		end
	else
		error('Please pass a valid dictonary struct or string\n');
	end
	
	if ~isempty(text)
		util.disp_section(sprintf('Dictionary %s', dict_string_or_struct));
		fprintf('%s\n', text);
		util.disp_section();
	end
	
	end