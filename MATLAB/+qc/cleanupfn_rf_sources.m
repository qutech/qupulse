function scan = cleanupfn_rf_sources(scan)
	
	if nargin < 1
		scan = [];
	end
	
	evalin('caller', 'cleanupFnRfMsg = onCleanup(@()(fprintf(''Executing cleanup function: Turned RF sources off\n'')));');
	evalin('caller', 'cleanupFnRf1 = onCleanup(@()(smset(''RF1_on'', 0)));');
	evalin('caller', 'cleanupFnRf2 = onCleanup(@()(smset(''RF2_on'', 0)));');
	
end