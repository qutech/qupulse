function setup_tek_awg(varargin)

global smdata
global plsdata

defaultArgs = struct( ...
  'sampleVoltPerAwgVolt', [util.db('dB2F',-45)*2 util.db('dB2F',-45)*2 util.db('dB2F',-45)*2 util.db('dB2F',-45)*2], ... % 10^(-dB/20)*ImpedanceMismatch
  'smChannels', {{'RFA', 'RFB', 'RFC', 'RFD'}}, ...
  'tekName', 'AWG5000', ...
  'globalTransformation', [], ...
  'ip', '169.254.40.80', ... %IP's:
  'dcMode', false, ...
  'maxPulseWait', 60 ... % Maximum waiting time in s in qc.awg_program before arming DAQ again
  );

args = util.parse_varargin(varargin, defaultArgs);
plsdata.awg.sampleVoltPerAwgVolt = args.sampleVoltPerAwgVolt;
plsdata.awg.dcMode = args.dcMode;
plsdata.awg.triggerStartTime = 0;
plsdata.awg.maxPulseWait = args.maxPulseWait;
plsdata.awg.minSamples = 250;
plsdata.awg.sampleQuantum = 1;
plsdata.awg.globalTransformation = args.globalTransformation;

for k = 1:numel(args.smChannels)
  smChannel = args.smChannels(k);
  if ~(smdata.channels(smchanlookup(smChannel)).instchan(1) == sminstlookup(args.tekName))
    error('Channel %s does not belong to %s\n', smChannel, args.tekName);
  end
  smdata.channels(smchanlookup(smChannel)).rangeramp(end) = 1/args.sampleVoltPerAwgVolt(k);
end

% Reload qctoolkit TEK AWG integration
qctoolkit_tek = py.importlib.reload(py.importlib.import_module('qctoolkit.hardware.awgs.tektronix'));

py.importlib.import_module('tek_awg');

awg = py.tek_awg.TekAwg.connect_to_ip('169.254.40.80');
awg.instrument.timeout = py.int(1200000);
% awg = py.tek_awg.TekAwg.connect_raw_visa_socket('169.254.40.80', '4001','@ni');
% % 
tawg=qctoolkit_tek.TektronixAWG(awg, pyargs('synchronize','clear'));
% Only real instrument
smdata.inst(sminstlookup(args.tekName)).data.tawg = tawg;

plsdata.awg.inst = smdata.inst(sminstlookup(args.tekName)).data.tawg;
if exist('awgctrl_tek.m', 'file')
  awgctrl('off')
end

% Create hardware setup for qctoolkit integration
plsdata.awg.hardwareSetup = py.qctoolkit.hardware.setup.HardwareSetup();

% Create python lambda function in Matlab
numpy = py.importlib.import_module('numpy');
for k = 1:numel(args.sampleVoltPerAwgVolt)
  multiply{k} = py.functools.partial(numpy.multiply, double(1./(args.sampleVoltPerAwgVolt(k))));
end

% PlaybackChannels can take more than two values (analog channels)
plsdata.awg.hardwareSetup.set_channel('TEK_A', ...
  py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst, int64(0), multiply{1}));
plsdata.awg.hardwareSetup.set_channel('TEK_B', ...
  py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst, int64(1), multiply{2}));
plsdata.awg.hardwareSetup.set_channel('TEK_C', ...
  py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst, int64(2), multiply{3}));
plsdata.awg.hardwareSetup.set_channel('TEK_D',  ...
  py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst, int64(3), multiply{4}));

% MarkerChannel can only take on two values (digital channels)
plsdata.awg.hardwareSetup.set_channel('TEK_A_MARKER', ...
  py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst, int64(0)));
plsdata.awg.hardwareSetup.set_channel('TEK_B_MARKER', ...
  py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst, int64(1)));
plsdata.awg.hardwareSetup.set_channel('TEK_C_MARKER', ...
  py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst, int64(2)));
plsdata.awg.hardwareSetup.set_channel('TEK_D_MARKER', ...
  py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst, int64(3)));
end