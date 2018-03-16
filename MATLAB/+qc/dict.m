function d = dict(varargin)
% Wrapper for struct so do not need to have three curly braces when
% creating pulse templates
%
% varargin needs to the same as for struct(.)

for k = 2:2:numel(varargin)
	varargin{k} = {varargin{k}};
end
d = struct(varargin{:});