function varargout = strrep(varargin)
	% Multiple string replacements:
	%   varargin{k} is a cell which should look like
	%   {cell of original strings, search string, replacement string, search string, replacement string, ...}
	%
	% If n arguments are received, n outputs are returned.
		
	for k = 1:numel(varargin)		
		varargout{k} = varargin{k}{1};		
		for l = 2:2:numel(varargin{k})
			varargout{k} = cellfun(@(x)(strrep(x, varargin{k}{l}, varargin{k}{l+1})), varargout{k}, 'UniformOutput', false);
		end		
	end	
	
end