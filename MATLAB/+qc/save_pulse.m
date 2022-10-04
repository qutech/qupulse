function [ file_written ] = save_pulse( pulse_template, overwrite )
	
	global plsdata
	
	if nargin < 2 || isempty(overwrite)
		overwrite = true;
	end
	
	file_written = false;
	
    if py.operator.contains(plsdata.qc.pulse_storage, pulse_template.identifier)
        if overwrite
            py.operator.delitem(plsdata.qc.pulse_storage, pulse_template.identifier);
        else
            warning('Did not write file as it exists and overwrite == false');
            return;
        end
    end
    
    try
        plsdata.qc.pulse_storage{pulse_template.identifier} = pulse_template;
        file_written = true;
% 		fprintf('File(s) written\n');
    catch err
        warning(err.getReport());
    end
end
	
	
	
