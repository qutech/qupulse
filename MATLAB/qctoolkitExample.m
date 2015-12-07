clear

% Setup temporary plsdata
global plsdata;
plsdata = [];
plsdata.datafile = [tempdir 'hardwaretest\plsdata_hw'];
plsdata.grpdir = [tempdir 'hardwaretest\plsdef\plsgrp'];
try
    rmdir([tempdir 'hardwaretest'],'s');
end
mkdir(plsdata.grpdir);

plsdata.pulses = struct('data', {}, 'name', {},	'xval',{}, 'taurc',{}, 'pardef',{},'trafofn',{},'format',{});
plsdata.tbase = 1000;

% Define some TablePulseTemplates
table_pulse_1 = py.qctoolkit.pulses.TablePulseTemplate();
table_pulse_1.add_entry('foo', 10);
table_pulse_1.add_entry(100, 0);

table_pulse_2 = py.qctoolkit.pulses.TablePulseTemplate();
table_pulse_2.add_entry(25, -10);
table_pulse_2.add_entry(50, 0);


% Build a sequence of TablePulseTemplates with given parameters
sequencer = py.qctoolkit.pulses.Sequencer();

clear parameters;
parameters.foo = 45;

sequencer.push(table_pulse_1, parameters);
sequencer.push(table_pulse_2);
parameters.foo = 80;
sequencer.push(table_pulse_1, parameters);
sequencer.push(table_pulse_1, parameters);

block = sequencer.build();
sequence = block.compile_sequence();

% Convert the instruction sequence to pulse_control pulses and a
% pulse_group
pci = py.qctoolkit.qcmatlab.pulse_control.PulseControlInterface(1e3);
qct_output = pci.create_pulse_group(sequence, 'foo');

pulse_group = convert_qctoolkit(qct_output);