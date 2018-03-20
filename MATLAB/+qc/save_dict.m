function save_dict(dict_struct)
	global plsdata
	delim = '___';
	
	if isstruct(dict_struct)	
		dict_struct = qc.array2list(dict_struct);
		text = py.json.dumps(dict_struct, pyargs('indent', int8(4), 'sort_keys', true));
		text = char(text);
		fileId = fopen(fullfile(plsdata.dict.path, [dict_struct.(strcat('dict', delim, 'name')) '.json']), 'w');
		fprintf(fileId, '%s', text);
		fclose(fileId);
	else
		error('Saving of dictionary failed since no struct was passed\n');
	end
end