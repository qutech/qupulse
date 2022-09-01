function scan = cleanupfn_rf_switches(scan)
	
	if nargin < 1
		scan = [];
	end
	
	evalin('caller', 'cleanupFnRfMsg = onCleanup(@()(fprintf(''Executing cleanup function: Turned RF Switches off\n'')));');
	evalin('caller', 'cleanupFnRf1 = onCleanup(@()(smset(''AWGSwitch1'', Inf)));');
	evalin('caller', 'cleanupFnRf2 = onCleanup(@()(smset(''AWGSwitch2'', Inf)));');
	
end