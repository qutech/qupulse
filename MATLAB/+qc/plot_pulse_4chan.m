function [t, channels, measurements] = plot_pulse_4chan(pulse, varargin)
% PLOT_PULSE_4CHAN Wrapper for plot_pulse specific for plotting pulses for
% two qubits with two control channels each
%
% (c) 2018/06 Pascal Cerfontaine (cerfontaine@physik.rwth-aachen.de)

defaultArgs = struct(...
	'charge_diagram_data_structs', {{}}, ... Should contain 2 structs in a cell array with fields x, y 
	                                     ... and data, where data{1} contains the charge diagram data
  'lead_points_cell',						 {{}}, ... Should contain a cell with a lead_points entry for each qubit
	'special_points_cell',			   {{}}, ... Should contain a cell with a special_points entry for each qubit
	'channels', {{'W', 'X', 'Y', 'Z'}}   ...
	);
args = util.parse_varargin(varargin, defaultArgs);
	
	
	for chrgInd = 1:2
		k = chrgInd + double(chrgInd==2);
		q = 4 - k;
		
		if numel(args.charge_diagram_data_structs) >= chrgInd
			args.charge_diagram_data = args.charge_diagram_data_structs{chrgInd};
			args.charge_diagram_data = {args.charge_diagram_data.x, args.charge_diagram_data.y, args.charge_diagram_data.data{1}};
		else
			args.charge_diagram_data = {};
		end
		
		if numel(args.lead_points_cell) >= chrgInd
			args.lead_points = args.lead_points_cell{chrgInd};
			else
			args.lead_points = {};
		end
		
		if numel(args.special_points_cell) >= chrgInd
			args.special_points = args.special_points_cell{chrgInd};
			else
			args.special_points = {};
		end
		
		args.charge_diagram = args.channels(k:k+1);
		args.subplots = [220+k 220+k+1];
		args.clear_fig = k==1;
		
		[t, channels, measurements] = qc.plot_pulse(pulse, args);			
		xlabel(args.channels(k));
		ylabel(args.channels(k+1));
		
		subplot(args.subplots(1));
		set(findall(gca, 'DisplayName', sprintf('Chan: %s', args.channels{q})), 'Visible', 'off');
		set(findall(gca, 'DisplayName', sprintf('Chan: %s', args.channels{q+1})), 'Visible', 'off');
		
	end
	
end