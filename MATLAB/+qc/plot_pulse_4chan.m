function [t, channels, measurements, instantiatedPulse] = plot_pulse_4chan(pulse, varargin)
% PLOT_PULSE_4CHAN Wrapper for plot_pulse specific for plotting pulses for
% two qubits with two control channels each
%
% (c) 2018/06 Pascal Cerfontaine (cerfontaine@physik.rwth-aachen.de)

defaultArgs = struct(...
	'charge_diagram_data_structs', {{}},								   ... Should contain 2 structs in a cell array with fields x, y 
																											   ... and data, where data{1} contains the charge diagram data
  'plot_charge_diagram',         true,								   ...
  'lead_points_cell',						 {{}},								   ... Should contain a cell with a lead_points entry for each qubit
	'special_points_cell',			   {{}},								   ... Should contain a cell with a special_points entry for each qubit
	'channels',										 {{'W', 'X', 'Y', 'Z'}}, ...
	'measurements',								 {{'A', 'A', 'B', 'B'}}, ...
	'markerChannels',							 {{'M1', '', 'M2', ''}}	 ...
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
		if args.plot_charge_diagram
			args.subplots = [220+k 220+k+1];
		else
			args.subplots = [210+chrgInd];
		end
		args.clear_fig = k==1;
		[t, channels, measurements, instantiatedPulse] = qc.plot_pulse(pulse, args);			
		xlabel(args.channels(k));
		ylabel(args.channels(k+1));
		
		if args.plot_charge_diagram
			subplot(args.subplots(1));
		end
		set(findall(gca, 'DisplayName', sprintf('Chan: %s', args.channels{q})), 'Visible', 'off');
		set(findall(gca, 'DisplayName', sprintf('Chan: %s', args.channels{q+1})), 'Visible', 'off');
		set(findall(gca, 'DisplayName', sprintf('Chan: %s', args.markerChannels{q})), 'Visible', 'off');
		set(findall(gca, 'DisplayName', sprintf('Chan: %s', args.markerChannels{q+1})), 'Visible', 'off');
		set(findall(gca, 'DisplayName', sprintf('Meas: %s', args.measurements{q})), 'Visible', 'off');		
		set(findall(gca, 'DisplayName', sprintf('Meas: %s', args.measurements{q+1})), 'Visible', 'off');	
		
		[hLeg, hObj] = legend(gca);
		for l = 1:numel(hLeg.String)
			if strcmp(hLeg.String{l}, sprintf('Chan: %s', args.channels{q})) || ...
				 strcmp(hLeg.String{l}, sprintf('Chan: %s', args.channels{q+1})) || ...
				 strcmp(hLeg.String{l}, sprintf('Chan: %s', args.markerChannels{q})) || ...
				 strcmp(hLeg.String{l}, sprintf('Chan: %s', args.markerChannels{q+1})) || ...
				 strcmp(hLeg.String{l}, sprintf('Meas: %s', args.measurements{q})) || ...
				 strcmp(hLeg.String{l}, sprintf('Meas: %s', args.measurements{q+1}))
			hLeg.String{l} = '';
		end
		findobj(hObj, 'type', 'line');
		set(hObj, 'lineWidth', 2);
	end
	
		
	
end