README path file for qc.qctoolkitTestSetup
------------------------------------------
written by M. Meyer 10|2018


Run the following code after inserting paths to create a path file. The generated file should be on the git ignore list.


%% ----------------------------
a = struct();
a.pulses_repo = '';
a.dicts_repo = '';
a.loadPath = '';
a.tunePath = '';
a.loadFile = '';
a.taborDriverPath = '';

quPulsePath = '...\qc-toolkit\MATLAB\+qc';

personalPathsStruct = a;
save([pathFile_save_path '\personalPaths'], personalPathsStruct);
%% -----------------------------