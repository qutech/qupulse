function scan = conf_seq_procfn(scan)
	% Dynamically changes the procfn arguments for each operation which
	% extracts the data from the channel ATSV
	%
	% Assumes that the field scan.data.daq_operations_length has been set to
	% the lengths of the operations.
	
	nGetChan = numel(scan.loops(1).getchan);
	lengths = scan.data.daq_operations_length;
	startInd = 1;
	for p = 1:numel(lengths)		
		scan.loops(1).procfn(nGetChan + p).fn(1).args = {startInd, startInd+lengths(p)-1};
		scan.loops(1).procfn(nGetChan + p).dim = [lengths(p)];
		startInd = startInd + lengths(p);
	end