function S = join_structs(varargin)
	% Fields of arguments passed in more on the left get overwritten by
	% arguments passed in more to the right
	
	S = struct();
	for v = varargin
		S = join_2_structs(S, v{1});
	end
	
end

function A = join_2_structs(A, B)
	% Fields in B are added to A, fields in A with same name get overwritten
	for f = fieldnames(B).'
		A.(f{1}) = B.(f{1});
	end
end