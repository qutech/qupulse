function pulseTemplate = struct_to_pulse(pulseStruct)
	
	backend = py.qctoolkit.serialization.DictBackend();
	serializer = py.qctoolkit.serialization.Serializer(backend);
	
	if startsWith(pulseStruct.main, '{')
		pulseName = 'main';
	else
		pulseName = pulseStruct.main;
	end
	
	backend.storage.update(pulseStruct)
	
	pulseTemplate = serializer.deserialize(pulseName);
	% 	plsStruct = util.py.py2mat(backend.storage)