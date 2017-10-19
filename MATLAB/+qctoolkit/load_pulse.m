function pulse_template = load_pulse(pulse_name, pulse_location)
	
backend = py.qctoolkit.serialization.FilesystemBackend(pulse_location);

serializer = py.qctoolkit.serialization.Serializer(backend);

pulse_template = serializer.deserialize(pulse_name);