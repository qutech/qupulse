function scan = cleanupfn_awg(scan)
	
	if nargin < 1
		scan = [];
	end
	
	evalin('caller', 'cleanupFnAwg = onCleanup(@()({awgctrl(''off''), fprintf(''Executing cleanup function: Turned AWG outputs off\n'')}));');
	
end