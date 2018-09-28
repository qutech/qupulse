function setup_tabor_awg(varargin)
	
	global smdata
	global plsdata
	
	defaultArgs = struct( ...
		'realAWG', true, ...
		'sampleVoltPerAwgVolt', [util.db('dB2F',-48)*2 util.db('dB2F',-48)*2 util.db('dB2F',-44)*2 util.db('dB2F',-48)*2], ... % 10^(-dB/20)*ImpedanceMismatch
		'simulateAWG', true, ...
		'smChannels', {{'RFA', 'RFB', 'RFC', 'RFD'}}, ...
		'taborName', 'TaborAWG2184C', ...
		'globalTransformation', [], ...
		'ip', '169.254.40.2', ...
		'dcMode', false, ...
		'maxPulseWait', 60, ... % Maximum waiting time in s in qc.awg_program before arming DAQ again
		'taborDriverPath', 'C:\Users\lablocal\Documents\PYTHON\TaborDriver\' ...
		);	
	args = util.parse_varargin(varargin, defaultArgs);
	plsdata.awg.sampleVoltPerAwgVolt = args.sampleVoltPerAwgVolt;
	plsdata.awg.dcMode = args.dcMode;
	plsdata.awg.triggerStartTime = 0;
	plsdata.awg.maxPulseWait = args.maxPulseWait;
	plsdata.awg.minSamples = 192;
	plsdata.awg.sampleQuantum = 16;
	plsdata.awg.globalTransformation = args.globalTransformation;
	
	for k = 1:numel(args.smChannels)
		smChannel = args.smChannels(k);
		if ~(smdata.channels(smchanlookup(smChannel)).instchan(1) == sminstlookup(args.taborName))
			error('Channel %s does not belong to %s\n', smChannel, args.taborName);
		end
		smdata.channels(smchanlookup(smChannel)).rangeramp(end) = 1/args.sampleVoltPerAwgVolt(k);
	end
	
	% Reload qctoolkit tabor AWG integration
	qctoolkit_tabor = py.importlib.reload(py.importlib.import_module('qctoolkit.hardware.awgs.tabor'));
		
	% Start simulator
	if args.simulateAWG		
		if py.pytabor.open_session('127.0.0.1') == py.None
			dos([fullfile(args.taborDriverPath, 'WX2184C.exe') ' /switch-on /gui-in-tray&'])
			
			while py.pytabor.open_session('127.0.0.1') == py.None
				pause(1);
				disp('Waiting for Simulator to start...');
			end
			disp('Simulator started');
		end
	end
	
	if args.realAWG && ~args.simulateAWG
		% Only real instrument
		smdata.inst(sminstlookup(args.taborName)).data.tawg = qctoolkit_tabor.TaborAWGRepresentation(['TCPIP::' args.ip '::5025::SOCKET'], pyargs('reset', py.True));
		smdata.inst(sminstlookup(args.taborName)).data.tawg.paranoia_level = int64(2);
	elseif args.realAWG && args.simulateAWG
		% Simulator and real instrument
	  smdata.inst(sminstlookup(args.taborName)).data.tawg = qctoolkit_tabor.TaborAWGRepresentation(['TCPIP::' args.ip '::5025::SOCKET'], pyargs('reset', py.True, 'mirror_addresses', {'127.0.0.1'}));
	elseif ~args.realAWG && args.simulateAWG
		% Just simulator
		smdata.inst(sminstlookup(args.taborName)).data.tawg = qctoolkit_tabor.TaborAWGRepresentation('TCPIP::127.0.0.1::5025::SOCKET', pyargs('reset', py.True));
	end
	
	plsdata.awg.inst = smdata.inst(sminstlookup(args.taborName)).data.tawg;
	if args.realAWG && exist('awgctrl.m', 'file')
		awgctrl('off');
	end
	
	% Create hardware setup for qctoolkit integration	
	plsdata.awg.hardwareSetup = py.qctoolkit.hardware.setup.HardwareSetup();	
	
	% Create python lambda function in Matlab
	numpy = py.importlib.import_module('numpy');
	for k = 1:numel(args.sampleVoltPerAwgVolt)
		multiply{k} = py.functools.partial(numpy.multiply, double(1./(args.sampleVoltPerAwgVolt(k))));
	end
	
	if args.realAWG || args.simulateAWG
		% PlaybackChannels can take more than two values (analog channels)
		plsdata.awg.hardwareSetup.set_channel('TABOR_A', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(0), multiply{1}));
		plsdata.awg.hardwareSetup.set_channel('TABOR_B', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(1), multiply{2}));
		plsdata.awg.hardwareSetup.set_channel('TABOR_C', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(0), multiply{3}));
		plsdata.awg.hardwareSetup.set_channel('TABOR_D',  ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(1), multiply{4}));
		
		plsdata.awg.hardwareSetup.set_channel('TABOR_AB', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(0), multiply{1}), py.True);
		plsdata.awg.hardwareSetup.set_channel('TABOR_AB', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(1), multiply{2}), py.True);
		
		
		plsdata.awg.hardwareSetup.set_channel('TABOR_AC', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(0), multiply{1}), py.True);
		plsdata.awg.hardwareSetup.set_channel('TABOR_AC', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(0), multiply{3}), py.True);
		
		plsdata.awg.hardwareSetup.set_channel('TABOR_AD', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(0), multiply{1}), py.True);
		plsdata.awg.hardwareSetup.set_channel('TABOR_AD', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(1), multiply{4}), py.True);
		
		plsdata.awg.hardwareSetup.set_channel('TABOR_BC', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(1), multiply{2}), py.True);
		plsdata.awg.hardwareSetup.set_channel('TABOR_BC', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(0), multiply{3}), py.True);
		
		plsdata.awg.hardwareSetup.set_channel('TABOR_BD', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(1), multiply{2}), py.True);
		plsdata.awg.hardwareSetup.set_channel('TABOR_BD', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(1), multiply{4}), py.True);
		
		plsdata.awg.hardwareSetup.set_channel('TABOR_CD', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(0), multiply{3}), py.True);
		plsdata.awg.hardwareSetup.set_channel('TABOR_CD', ...
			py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(1), multiply{4}), py.True);
		
		
		% MarkerChannel can only take on two values (digital channels)
		plsdata.awg.hardwareSetup.set_channel('TABOR_A_MARKER', ...
			py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst.channel_pair_AB, int64(0)));
		plsdata.awg.hardwareSetup.set_channel('TABOR_B_MARKER', ...
			py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst.channel_pair_AB, int64(1)));
		plsdata.awg.hardwareSetup.set_channel('TABOR_C_MARKER', ...
			py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst.channel_pair_CD, int64(0)));
		plsdata.awg.hardwareSetup.set_channel('TABOR_D_MARKER', ...
			py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst.channel_pair_CD, int64(1)));
	end