function [W, X, Y, Z] = compensate_channels(t, W, X, Y, Z, interpolation)
% ADD_CHANNELS Compensate linear crosstalk for two ST0 qubits
%   This function adds W and X to Y and Z (and vice-versa) with individual
%   multipliers. It is specifically designed for compensating linear control
%   crosstalk two ST0 qubits but might be suited for other qubit
%   implementations as well.
%
% --- Outputs -------------------------------------------------------------
% W, X           : Compensated qubit 1 channel values (cell)
% Y, Z           : Compensated qubit 2 channel values (cell)
%
% --- Inputs --------------------------------------------------------------
% t, W, X, Y, Z, interpolation, MTrig, M1, M2 are cells with the same
% number of entries
%
% t              : Time indices of the channel values (cell)
% W, X           : Qubit 1 channel values (cell)
% Y, Z           : Qubit 2 channel values (cell)
% interpolation  : Interpolation strategies, use ‘hold’ for default
%                  qc-toolkit behaviour
%
% -------------------------------------------------------------------------
% (c) 2018/05 Pascal Cerfontaine (cerfontaine@physik.rwth-aachen.de)

	assert(all(numel(t) == cellfun(@numel, {W, X, Y, Z, interpolation})), 'Cell input arguments must have the same number of elements');

	[W, X, Y, Z] = comp_channels(W, X, Y, Z);	
	[W, X, Y, Z] = format_channels(t, interpolation, W, X, Y, Z);

end


function [W, X, Y, Z] = comp_channels(W, X, Y, Z)		
	for k = 1:numel(W)
		Wk = W{k};
		Xk = X{k};
		
		W{k} = [W{k} '   + comp_w_y*(  ' Y{k} '  )   + comp_w_z*(  ' Z{k} '  )'];
		X{k} = [X{k} '   + comp_x_y*(  ' Y{k} '  )   + comp_x_z*(  ' Z{k} '  )'];
		
		Y{k} = [Y{k} '   + comp_y_w*(  '  Wk  '  )   + comp_y_x*(  '  Xk  '  )'];
		Z{k} = [Z{k} '   + comp_z_w*(  '  Wk  '  )   + comp_z_x*(  '  Xk  '  )'];		
	end
	
end

function varargout = format_channels(t, interpolation, varargin)
	
	for k = 1:numel(t)
		for v = 1:numel(varargin)
			
			varargout{v}{end+1} = { t{k}, varargin{v}{k}, interpolation{k} };
			
		end		
	end
	
end