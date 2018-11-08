function workaround_alazar_single_buffer_acquisition()
	% Some workaround for a workaround - ask Simon
	global plsdata
	plsdata.daq.inst.config.totalRecordSize = int64(0);
% 	plsdata.daq.inst.config.aimedBufferSize = int64(2^24);
	plsdata.daq.inst.config.aimedBufferSize = int64(2^20);
	plsdata.daq.inst.card.reset
	plsdata.daq.inst.update_settings = py.True;
	
	fprintf('qc.workaround_alazar_single_buffer_acquisition executed\n');