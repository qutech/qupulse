function [analogNames, markerNames, channels] = get_awg_channels()
	
	global plsdata
	
	% Get AWG analog channels and markers
	channels = struct(plsdata.awg.hardwareSetup.registered_channels);		
	analogNames = {};
	markerNames = {};
	for chanName = fieldnames(channels)'
	  chan = util.py.py2mat(channels.(chanName{1}));
		if isa(chan{1}, 'py.qctoolkit.hardware.setup.MarkerChannel')
			markerNames{end+1} = chanName{1};
		elseif isa(chan{1}, 'py.qctoolkit.hardware.setup.PlaybackChannel')
			analogNames{end+1} = chanName{1};
		end
	end