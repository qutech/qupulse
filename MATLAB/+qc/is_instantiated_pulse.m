function bool = is_instantiated_pulse(pulse)
	bool = strcmp(class(pulse), 'py.qctoolkit._program.instructions.ImmutableInstructionBlock');
    bool = bool || strcmp(class(pulse), 'py.qctoolkit._program._loop.Loop');
