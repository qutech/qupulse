%% --- Test setup without AWG and Alazar (only qctoolkit) -----------------
global plsdata
plsdata = struct( ...
	'path', 'C:\Users\Marcel Meyer\Documents\__Wissenschaft__\Master_Projekt\code\qctk_pulses', ...
	'awg', struct('inst', [], 'hardwareSetup', [], 'sampleRate', 2e9, 'currentProgam', '', 'registeredPrograms', struct(), 'defaultChannelMapping', struct(), 'defaultWindowMapping', struct(), 'defaultParametersAndDicts', {{}}, 'defaultAddMarker', {{}}), ...
  'dict', struct('cache', [], 'path', 'C:\Users\Marcel Meyer\Documents\__Wissenschaft__\Master_Projekt\code\qctk_dicts'), ...
	'qc', struct('figId', 801), ...
	'daq', struct('inst', [], 'defaultOperations', {{}}, 'reuseOperations', false) ...
	);
plsdata.daq.instSmName = 'ATS9440Python';
plsdata.qc.backend = py.qctoolkit.serialization.FilesystemBackend(plsdata.path);
plsdata.qc.serializer = py.qctoolkit.serialization.Serializer(plsdata.qc.backend);
disp('[second script (plsdata) done]');
% -------------------------------------------------------------------------

%% --- Test setup replicating the Triton 200 measurement setup ------------
% Does not replicate Alazar functionality as there is no simulator
% Need the triton_200 repo on the path (for awgctrl)

% Path for Triton 200 backups
loadPath = 'C:\Users\Marcel Meyer\Documents\__Wissenschaft__\Master_Projekt\copy_from_trition200_backup\DATA\workspace';
pulsePath = plsdata.path;
dictPath = plsdata.dict.path;
tunePath = 'C:\Users\Marcel Meyer\Documents\__Wissenschaft__\Master_Projekt\DATA\workspace';

% Loading
try
	copyfile(fullfile(loadPath, 'smdata_recent.mat'), fullfile(tunePath, 'smdata_recent.mat'));
catch err
	warning(err.getReport());
end
load(fullfile(tunePath, 'smdata_recent.mat'));
info = dir(fullfile(tunePath, 'smdata_recent.mat'));
fprintf('Loaded smdata from %s\n', datestr(info.datenum));
try
  copyfile(fullfile(loadPath, 'tunedata_recent.mat'), fullfile(tunePath, 'tunedata_recent.mat'));
% 	copyfile(fullfile(loadPath, 'tunedata_2018_06_25_21_08.mat'), fullfile(tunePath, 'tunedata_recent.mat'));
catch err
	warning(err.getReport());
end
load(fullfile(tunePath, 'tunedata_recent.mat'));
info = dir(fullfile(tunePath, 'tunedata_recent.mat'));
fprintf('Loaded tunedata from %s\n', datestr(info.datenum));

try
	copyfile(fullfile(loadPath, 'plsdata_recent.mat'), fullfile(tunePath, 'plsdata_recent.mat'));
catch err
	warning(err.getReport());
end
load(fullfile(tunePath, 'plsdata_recent.mat'));
info = dir(fullfile(tunePath, 'plsdata_recent.mat'));
fprintf('Loaded plsdata from %s\n', datestr(info.datenum));

global tunedata
global plsdata
tunedata.run{tunedata.runIndex}.opts.loadFile = 'X:\Common\GaAs\Triton 200\Backup\DATA\tune';
import tune.tune

plsdata.path = pulsePath;

% Alazar dummy instrument (simulator not implemented yet)
smdata.inst(sminstlookup(plsdata.daq.instSmName)).data.address = 'simulator';
plsdata.daq.inst = py.qctoolkit.hardware.dacs.alazar.AlazarCard([]);

% Setup AWG
% Turns on AWG for short time but turns it off again
% Initializes hardware setup
% Can also be used for deleting all programs/resetting but then also need to setup Alazar again, i.e. the cell above and the three cells below )
plsdata.awg.hardwareSetup = [];
qc.setup_tabor_awg('realAWG', false, 'simulateAWG', true, 'taborDriverPath', 'C:\Users\"Marcel Meyer"\Documents\__Wissenschaft__\Master_Projekt\code\python-TaborDriver');

% AWG default settings
awgctrl('default');
plsdata.awg.inst.send_cmd(':OUTP:COUP:ALL HV');

% Alazar
% Execute after setting up the AWG since needs hardware setup initialized
% Need to test whether need to restart Matlab if execute
% qc.setup_alazar_measurements twice
qc.setup_alazar_measurements('nQubits', 2, 'nMeasPerQubit', 4, 'disp', true);

% Qctoolkit
plsdata.qc.backend = py.qctoolkit.serialization.FilesystemBackend(pulsePath);
plsdata.qc.serializer = py.qctoolkit.serialization.Serializer(plsdata.qc.backend);
plsdata.dict.path = dictPath;

% Tune
tunedata.run{tunedata.runIndex}.opts.path = tunePath;


import tune.tune
disp('[third script (awg, tunedata etc) done]');
% -------------------------------------------------------------------------