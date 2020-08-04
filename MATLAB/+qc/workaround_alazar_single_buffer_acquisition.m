function workaround_alazar_single_buffer_acquisition()
	% the alazar acquisition might fail if there is just a single buffer
	% the functionality to work around that was moved to the attribute
	% 'buffer_strategy' of the qupulse driver object
	
	% Some workaround for a workaround - ask Simon
	global plsdata
	plsdata.daq.inst.config.totalRecordSize = int64(0);
% 	plsdata.daq.inst.config.aimedBufferSize = int64(2^24);


if true
	fprintf('qc.workaround_alazar_single_buffer_acquisition decidete to try one buffer per measurement window\n');
	plsdata.daq.inst.buffer_strategy = py.qupulse.hardware.dacs.alazar.OneBufferPerWindow();
else
	% use default behaviour
	plsdata.daq.inst.buffer_strategy = py.None;
end

	plsdata.daq.inst.card.reset
	plsdata.daq.inst.update_settings = py.True;
	
	fprintf('qc.workaround_alazar_single_buffer_acquisition executed\n');