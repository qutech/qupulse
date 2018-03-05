function scan = cleanupfn_delete_getchans(scan, getchans)
	
	for getchan = getchans
		evalin('caller', sprintf('data{%i} = {};', getchan));
	end
	
end