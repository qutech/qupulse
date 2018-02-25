function bool = is_instantiated_pulse(pulse)
	bool = strcmp(class(pulse), 'py.qctoolkit.pulses.instructions.ImmutableInstructionBlock');