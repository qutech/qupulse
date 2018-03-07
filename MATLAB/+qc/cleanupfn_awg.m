function scan = cleanupfn_awg(scan)
	
	if nargin < 1
		scan = [];
	end
	
	evalin('caller', 'cleanupFn1 = onCleanup(@()(fprintf(''Executing cleanup function:\n'')));');
	evalin('caller', 'cleanupFn2 = onCleanup(@()(awgctrl(''off'')));');
	
end