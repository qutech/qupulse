function resolved = resolve_mappings(varargin)
% Takes a variable number of parameter mappings (structs) and resolves them
% sequentially. Eg. if the first maps A to B and the second B to C, returns
% a mapping A to C. Goes left-to-right (first to last).
%
% >> qc.resolve_mappings(struct('a', 'b'), struct('b', 'c', 'x', 'y'), ...
%                        struct('c', 'd'), struct('x', 'z', 'asdf', 'jkl'), ...
%                        struct('z', 'a', 'bla', 'foo'))
% 
% Warning: Field clash. Value to the right supercedes. 
% > In qc.resolve_mappings (line 14) 
% 
% ans = 
% 
%   struct with fields:
% 
%        a: 'd'
%        x: 'a'
%     asdf: 'jkl'
%      bla: 'foo'
% =========================================================================
resolved = varargin{1};
varargin = varargin(2:end);

while ~isempty(varargin)
    visited_fields = {};
    for f = fieldnames(resolved)'
        field = f{1};
        value = resolved.(field);
        if isfield(varargin{1}, field)
            warning('Field clash. Value to the right supercedes.')
            resolved.(field) = varargin{1}.(field);
            visited_fields = [visited_fields field];
        elseif isfield(varargin{1}, value)
            resolved.(field) = varargin{1}.(value);
            visited_fields = [visited_fields value];
        end
    end
    
    % Add unchanged new fields from varargin{1}
    for f = fieldnames(varargin{1})'
        field = f{1};
        value = varargin{1}.(field);
        if ~ismember(visited_fields, field)
            resolved.(field) = value;
        end
    end
    
    varargin = varargin(2:end);
end

end