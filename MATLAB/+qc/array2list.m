function sOut = array2list(sIn)
	if isstruct(sIn)
		sOut = structfun(@qc.array2list, sIn, 'UniformOutput', false);
	elseif isnumeric(sIn) && ~isscalar(sIn)
		sOut = py.list(sIn(:).');
	else
		sOut = sIn;
	end
end