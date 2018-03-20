function sOut = array2row(sIn)
	if isstruct(sIn)
		sOut = structfun(@qc.array2row, sIn, 'UniformOutput', false);
	elseif isnumeric(sIn) && ~isscalar(sIn)
		sOut = sIn(:).';
	else
		sOut = sIn;
	end
end