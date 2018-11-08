function pulse_group = convert_qctoolkit(qct_output)
% pulse_group = convert_qctoolkit(qct_output)
% 
% Registers pulses and converts pulse group data obtained from qupulse.
%
% qct_output: The output tuple of the
% PulseControlInterface.create_pulse_group() method.

qct_pulses = qct_output{2};

% Convert Python dicts of pulses to pulse control waveform pulse structs
% and register them using plsreg. Remember index in pulse database.
pulse_indices = zeros(size(qct_pulses, 2));
for i = 1:size(qct_pulses, 2)
    pulse = struct(qct_pulses{i});
    pulse.name = arrayfun(@char, pulse.name);
    pulse.data = struct(pulse.data);
    pulse.data.marker = cell2mat(cell(pulse.data.marker));
    pulse.data.wf = cell2mat(cell(pulse.data.wf));
    pulse_indices(i) = plsreg(pulse);
end

% Convert Python dict of pulse group to pulse control struct.
% Replace pulse indices in pulse_group.pulses with the indices of the
% pulses in the pulse database (plsdata).
pulse_group = struct(qct_output{1});
pulse_group.chan = double(pulse_group.chan);
pulse_group.name = arrayfun(@char, pulse_group.name);
pulse_group.ctrl = arrayfun(@char, pulse_group.ctrl);
pulse_group.nrep = cellfun(@double, cell(pulse_group.nrep));
pulse_group.pulses = cellfun(@double, cell(pulse_group.pulses));
for i = 1:size(pulse_group.pulses, 2)
    pulse_group.pulses(i) = pulse_indices(pulse_group.pulses(i) + 1);
end