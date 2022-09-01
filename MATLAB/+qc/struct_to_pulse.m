function pulseTemplate = struct_to_pulse(pulseStruct)

	backend = py.qctoolkit.serialization.DictBackend();
	pulse_storage = py.qctoolkit.serialization.PulseStorage(backend);
	
    % THIS IS WRONG!!!
	if startsWith(pulseStruct.main, '{')
		pulseName = 'main';
	else
		pulseName = pulseStruct.main;
    end
	
    % feed into backend
	backend.storage.update(pulseStruct)
	
    
	pulseTemplate = pulse_storage.get(pulseName);
	% 	plsStruct = util.py.py2mat(backend.storage)