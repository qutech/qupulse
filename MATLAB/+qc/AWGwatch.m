function AWGwatch()
% starts a matlab (vers 2018a) app to DISPLAY SEQUENCER TABLES AND
% WAFEFORMS in qctoolkit and on the Tabor AWG simulatar
% -------------------------------------------------------------------------
% - to edit the app open awgdisp_app.mlapp in the Matlab app designer
% - user preferences can be edited in the app private properties directly
%   at the top in awgdisp_app.mlapp       
% -------------------------------------------------------------------------
% App written by Marcel Meyer 08|2018          marcel.meyer1@rwth-aachen.de

  disp('AWGwatch - app is started');
  
  % the app is not on path +qc because then one has problems debugging it
  pathOfApp = which('qc.AWGwatch');
  pathOfApp = pathOfApp(1:end-10);
  pathOfApp = [pathOfApp 'AWGwatch\'];
  addpath(pathOfApp);
	
  awgdisp_app();
	
end