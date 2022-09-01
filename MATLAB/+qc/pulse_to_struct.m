function pulseStruct = pulse_to_struct(pulseTemplate)	
	
	backend = py.qctoolkit.serialization.DictBackend();
	storage = py.qctoolkit.serialization.PulseStorage(backend);
    
    % THIS IS WRONG!
    storage.overwrite('main',pulseTemplate)
    
	pulseStruct = util.py.py2mat(backend.storage);
	
	if ~isfield(pulseStruct, 'main')
		pulseStruct.main = char(pulseTemplate.identifier);
	end
	
	

	