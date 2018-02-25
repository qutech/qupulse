function cmdout = git(varargin)
GITPATH = 'Y:\GaAs\Bethke\git_local\bin\git.exe';
options = '';
for i=1:numel(varargin)
    options = [options ' ' varargin{i}];
end
[status, cmdout] = system([GITPATH ' ' options]);
if status ~= 0
    error('Caught error in git subprocess: %s', cmdout);
end
end