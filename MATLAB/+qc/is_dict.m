function bool = is_dict(dp)
	delim = '___';	
	bool = ischar(dp) || (isstruct(dp) && isfield(dp, strcat('dict', delim, 'name')));
end