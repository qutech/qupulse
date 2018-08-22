function awgdisp(source)

if nargin < 1
  source = 'qctk';
end

% get AWG objects
switch source
  case 'qctk'
    disp('source = qctoolkit')
  case 'sim'
    disp('simulaotr')
  otherwise
    disp('no input')
end
    

% get sequence tables

% create app