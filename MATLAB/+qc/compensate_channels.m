function [W, X, Y, Z] = compensate_channels(t, W, X, Y, Z, interpolation, comp_param_name)
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
%                  One row:   Applied to all channels
%                  *Two rows:  Row 1 applied to W, X and row 2 to Y, Z
%                  *Four rows: One row for each channel
% W, X           : Qubit 1 channel values (cell)
% Y, Z           : Qubit 2 channel values (cell)
% interpolation  : Interpolation strategies, use ‘hold’ for default
%                  qupulse behaviour
%                  One row:   Applied to all channels
%                  *Two rows:  Row 1 applied to W, X and row 2 to Y, Z
%                  *Four rows: One row for each channel
% comp_param_name: Name of the compensation parameters
%
% * Currently disabled since compensation might not work if using
%   different times
%
% -------------------------------------------------------------------------
% (c) 2018/05 Pascal Cerfontaine (cerfontaine@physik.rwth-aachen.de)

	assert(all(numel(t) == cellfun(@numel, {W, X, Y, Z, interpolation})), 'Cell input arguments must have the same number of elements');
  assert(size(t, 1) == 1 && size(interpolation, 1) == 1, 'Compensation of different times and interpolation strategies not currently supported');
	
	if nargin < 7 || isempty(comp_param_name)
		comp_param_name = 'globals___comp';
	end
	
	if ~strcmp(comp_param_name, 'compensation_off')
		[W, X, Y, Z] = comp_channels(W, X, Y, Z, comp_param_name);
	end
	[W, X, Y, Z] = format_channels(t, interpolation, W, X, Y, Z);

end


function [W, X, Y, Z] = comp_channels(W, X, Y, Z, comp_param_name)		
	for k = 1:numel(W)
		Wk = W{k};
		Xk = X{k};
		
		W{k} = [W{k} '   + ' comp_param_name '_w_y*(  ' Y{k} '  )   + ' comp_param_name '_w_z*(  ' Z{k} '  )'];
		X{k} = [X{k} '   + ' comp_param_name '_x_y*(  ' Y{k} '  )   + ' comp_param_name '_x_z*(  ' Z{k} '  )'];
		
		Y{k} = [Y{k} '   + ' comp_param_name '_y_w*(  '  Wk  '  )   + ' comp_param_name '_y_x*(  '  Xk  '  )'];
		Z{k} = [Z{k} '   + ' comp_param_name '_z_w*(  '  Wk  '  )   + ' comp_param_name '_z_x*(  '  Xk  '  )'];		
	end
	
end

function varargout = format_channels(t, interpolation, varargin)
	
	varargout = cell(numel(varargin), 1);
	
	if size(interpolation, 1) == 1
		interpolation = repmat(interpolation, 4, 1);
	elseif size(interpolation, 1) == 2
		interpolation = [ repmat(interpolation(1, :), 2, 1) ;
											repmat(interpolation(2, :), 2, 1) ];
	end
	
	if size(t, 1) == 1
		t = repmat(t, 4, 1);
	elseif size(t, 1) == 2
		t = [ repmat(t(1, :), 2, 1) ;
					repmat(t(2, :), 2, 1) ];
	end
	
	for k = 1:size(t, 2)
		for v = 1:numel(varargin)
			
			varargout{v}{end+1} = { t{v, k}, varargin{v}{k}, interpolation{v, k} };
			
		end		
	end
	
end