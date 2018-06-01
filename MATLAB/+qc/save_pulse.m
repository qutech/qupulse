function [ file_written ] = save_pulse( pulse_template, overwrite )
	
	global plsdata
	
	if nargin < 2 || isempty(overwrite)
		overwrite = true;
	end
	
	file_written = false;
	
	try
		plsdata.qc.serializer.serialize(pyargs('serializable', pulse_template, 'overwrite', overwrite));
		file_written = true;
% 		fprintf('File(s) written\n');
	catch err
		if util.str_contains(err.message, 'FileExistsError')
			warning('%s\n', strrep(err.message, 'Python Error: ', ''));
		else
			warning(err.getReport());
		end
	end
	
	
	
