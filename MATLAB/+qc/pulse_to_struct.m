function pulseStruct = pulse_to_struct(pulseTemplate)	
	
	backend = py.qctoolkit.serialization.DictBackend();
	serializer = py.qctoolkit.serialization.Serializer(backend);
	
	serializer.serialize(pulseTemplate);
	pulseStruct = util.py.py2mat(backend.storage);
	
	if ~isfield(pulseStruct, 'main')
		pulseStruct.main = char(pulseTemplate.identifier);
	end
	
	

	