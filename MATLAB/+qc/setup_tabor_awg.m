function setup_tabor_awg(varargin)
	
	global smdata
	global plsdata
	
	defaultArgs = struct( ...
		'realAWG', true, ...
		'simulateAWG', true, ...
		'taborName', 'TaborAWG2184C', ...
		'ip', '169.254.40.2', ...
		'taborDriverPath', 'C:\Users\lablocal\Documents\PYTHON\TaborDriver\' ...
		);	
	args = util.parse_varargin(varargin, defaultArgs);
	
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
		smdata.inst(sminstlookup(args.taborName)).data.tawg = qctoolkit_tabor.TaborAWGRepresentation(['TCPIP::' a.ip '::5025::SOCKET'], pyargs('reset', py.True));
		smdata.inst(sminstlookup(args.taborName)).data.tawg.paranoia_level = int64(2);
	elseif args.realAWG && args.simulateAWG
		% Simulator and real instrument
	  smdata.inst(sminstlookup(args.taborName)).data.tawg = py.qctoolkit.hardware.awgs.tabor.TaborAWGRepresentation(tabor_address, pyargs('reset', py.True, 'mirror_addresses', {'127.0.0.1'}));
	elseif ~args.realAWG && args.simulateAWG
		% Just simulator
		smdata.inst(sminstlookup(args.taborName)).data.tawg = qctoolkit_tabor.TaborAWGRepresentation('TCPIP::127.0.0.1::5025::SOCKET', pyargs('reset', py.True));
	end
	
	plsdata.awg.inst = smdata.inst(sminstlookup(args.taborName)).data.tawg;
	
	% Create hardware setup for qctoolkit integration	
	plsdata.awg.hardwareSetup = py.qctoolkit.hardware.setup.HardwareSetup();
	
	% PlaybackChannels can take more than two values (analog channels)
	plsdata.awg.hardwareSetup.set_channel('TABOR_A', py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(0)));
	plsdata.awg.hardwareSetup.set_channel('TABOR_B', py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_AB, int64(1)));	
	plsdata.awg.hardwareSetup.set_channel('TABOR_C', py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(0)));
	plsdata.awg.hardwareSetup.set_channel('TABOR_D', py.qctoolkit.hardware.setup.PlaybackChannel(plsdata.awg.inst.channel_pair_CD, int64(1)));
	
	% MarkerChannel can only take on two values (digital channels)
	plsdata.awg.hardwareSetup.set_channel('TABOR_A_MARKER', py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst.channel_pair_AB, int64(0)));
	plsdata.awg.hardwareSetup.set_channel('TABOR_B_MARKER', py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst.channel_pair_AB, int64(1)));	
	plsdata.awg.hardwareSetup.set_channel('TABOR_C_MARKER', py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst.channel_pair_CD, int64(0)));
	plsdata.awg.hardwareSetup.set_channel('TABOR_D_MARKER', py.qctoolkit.hardware.setup.MarkerChannel(plsdata.awg.inst.channel_pair_CD, int64(1)));