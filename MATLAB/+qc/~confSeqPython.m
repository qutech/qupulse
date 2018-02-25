function scan = confSeqPython(plsgrp, nloop, nrep, varargin)
global awgdata
global smdata
global fbdata
inst_index = sminstlookup('ATS9440Python');

p = inputParser;
p.addRequired('plsgrp', @validate_plsgrp);
p.addRequired('nloop', @util.validation.isposint); % no complicated validation possible without npulse etc.
p.addRequired('nrep', @util.validation.isposint); % only check for positive integer
p.addOptional('useFb', false, @islogical);
p.addOptional('usePol', false, @islogical);
p.addOptional('operations', {'DS', 'RSA', 'HIST'}, @validate_operations);
p.addOptional('dispchan', 'default', @validate_dispchan);
p.addOptional('T', 4, @isnumeric);
p.addOptional('dsmask', 0, @isnumeric);
p.addOptional('histRange', [0e-3 20.0e-3], @isnumeric);
p.addOptional('genMask', false);
p.addOptional('genMaskOffset', 25, @isnumeric);
p.addOptional('forceMasks', false, @islogical);
p.addOptional('seqhackCompat', false, @islogical);
p.addOptional('inline', false, @validate_inline);

p.parse(plsgrp, nloop, nrep, varargin{:});

plsgrp = p.Results.plsgrp;
nloop = p.Results.nloop;
nrep = p.Results.nrep;
useFb = p.Results.useFb; 
usePol = p.Results.usePol;
ops = p.Results.operations;
dispchan = p.Results.dispchan;
T = p.Results.T; %pulselength in us
dsmask = p.Results.dsmask;
forcemasks = p.Results.forceMasks;
seqhackCompat = p.Results.seqhackCompat;
% inline = p.Results.inline;

histRange = p.Results.histRange;
genMask = p.Results.genMask;
genMaskOffset = p.Results.genMaskOffset;

if usePol
	useFb = true;
end

% if isstring(inline) || ischar(inline) || inline
% 	
% end

%% sanity check part

plsgrp = awggrpind(plsgrp);

oversamp = 1;
% nrep = 1;
sampler = nan;

npulse = awgdata.pulsegroups(plsgrp(1)).npulse(1);
dccomp = 0;

% check for smallest possible nloop
smallest_nloop = nloop_finder(T, npulse);
if mod(nloop, smallest_nloop) % implicit ~= 0
    suggestion = nloop + mod(nloop, smallest_nloop);
    error('For %d pulses with length %.3f us, nloop has to be a multiple of %d.\nI suggest using %d instead', npulse, T, smallest_nloop, suggestion)
end

%% scan building part
if genMask
    offset = genMaskOffset;
    plsgrp = awggrpind(plsgrp);
    names = {awgdata.pulsegroups(plsgrp).name};
    pg = plsmakegrp(names(1), 'common');
    [generated_mask, scanlinelength] = plsgenmask(pg, 'offset', offset, 'channel', 'A', 'nloop', nloop, 'seqhackMerge', seqhackCompat);
    for i=2:numel(names)
        pg = plsmakegrp(names(i), 'common');
        if ~isequal(generated_mask, plsgenmask(pg, 'offset', offset, 'nloop', nloop))
					  if ~forcemasks
            error('The masks of the first pulse group and pulse group %d/%d do not match', i, numel(names))
						else
							warning('The masks of the first pulse group and pulse group %d/%d do not match', i, numel(names))
						end
        end
    end
end
%% first prepare masks and operations, build alazar config
masks = {};
operations = {};
% operation downsampling over the measurement window
if ismember('DS', ops) || ismember('HIST', ops)
    if genMask
        masks = [masks generated_mask];
    elseif dsmask == 0 % use defaults
        masks{end+1}.type = 'Periodic Mask';
        measurement_window_index = size(masks, 2);
        masks{end}.begin = 170;
        masks{end}.begin = 250;
        masks{end}.end = T * 100 - 10;
        masks{end}.period = T * 100;
        masks{end}.channel = 'A';
    else
        masks{end+1}.type = 'Periodic Mask';
        measurement_window_index = size(masks, 2);
        masks{end}.begin = dsmask(1);
        masks{end}.end = dsmask(2);
        masks{end}.period = T * 100;
				masks{end}.channel = 'A';
    end
end

if ismember('DS', ops)
    if genMask
        for m=1:numel(masks)
            operations{end+1}.type = 'DS + REP AV';
            operations{end}.mask = m; % use mask defined above
						operations{end}.period = npulse;
						if strcmp(masks{m}.type, 'Auto Mask')
							operations{end}.period = numel(masks{m}.begin) / nloop;
						else
							operations{end}.period = npulse;
						end
        end
    else
        operations{end+1}.type = 'DS + REP AV';
        operations{end}.mask = measurement_window_index; % use mask defined above
        operations{end}.period = npulse;
    end
end

% repetitive signal averaging over the whole pulse
if ismember('RSA', ops)
    masks{end+1}.type = 'Periodic Mask';
    masks{end}.begin = 0;
    masks{end}.end = T * 100;
    masks{end}.period = T * 100;
    masks{end}.channel = 'A';

    operations{end+1}.type = 'REP AV';
    operations{end}.mask = size(masks, 2); % use mask defined above
end

% histogram over the measurement window
if ismember('HIST', ops)
    if genMask
        for m=1:numel(masks)
            operations{end+1}.type = 'HIST';
            operations{end}.mask = m;
            operations{end}.range_in_volts = histRange; %TODO: check
            operations{end}.bin_count = 500;
        end
    else
        operations{end+1}.type = 'HIST';
        operations{end}.mask = measurement_window_index;
        operations{end}.range_in_volts = histRange; %TODO: check
        operations{end}.bin_count = 500;
    end
end

% build config struct from defaults and scan specific values
config = AlazarDefaultSettings(inst_index);
config.masks = masks;
config.operations = operations;
if genMask
	config.total_record_size = scanlinelength;
else
	config.total_record_size = 100 * T * npulse * nloop;
end
% scan.data.config = config; % save config to scan

datachan = arrayfun(@(x) sprintf('ATS%d', x), 1:numel(operations), 'UniformOutput', 0);
% switch(numel(ops))
%     case 1
%         datachan = {'ATS1'};
%     case 2
%         datachan = {'ATS1', 'ATS2'};
%     case 3
%         datachan = {'ATS1', 'ATS2', 'ATS3'};
%     case 4
%         datachan = {'ATS1', 'ATS2', 'ATS3', 'ATS4'};
% end


auxchan = {};

% polarization parameters
offpos=[]; %use -2 or list to add off pulses if necessary ???TB
% pulselist=[awgseqind('polS200')  awgseqind('polT200') ];
% pulselist=[awgseqind('polS200')  awgseqind('polT200') awgseqind('fb_15_12') awgseqind('fb_15_13') awgseqind('fb_15_33') awgseqind('fb_15_21') awgseqind('fb_24_12') awgseqind('fb_24_13') awgseqind('fb_24_33') awgseqind('fb_24_21') awgseqind('fb_24_32')]; % awgseqind for S,T+,Sfeed;
if useFb
    pulselist = [awgseqind(fbdata.fbgroup)]; % awgseqind for S,T+,Sfeed;
else  
    pulselist = []; % awgseqind for S,T+,Sfeed;
end
% % pulselist = [awgseqind({pgroups{:}}) awgseqind(fbdata.fbgroup)]; % added edsr pulses;
pulsetab = repmat([1, Inf, nan, 1],length(pulselist),1);
pulsetab(:,3)=pulselist';
pulsetab(:,4)=1;
fbmode = 2;


%% create scan
plsgrp = awggrpind(plsgrp);
if plsgrp(1) > 0
    if isnan(npulse)
        npulse = awgdata.pulsegroups(plsgrp(1)).npulse(1);
    elseif npulse < 0
        npulse = awgdata.pulsegroups(plsgrp(1)).npulse(1) * abs(npulse);
    end

    for ii = 1:length(plsgrp)
       if(npulse ~= awgdata.pulsegroups(plsgrp(ii)).npulse(1))
           warning(sprintf('Pulse number mismatch; length of group %s (%d) is not %d\n',...
               awgdata.pulsegroups(plsgrp(ii)).name,awgdata.pulsegroups(plsgrp(ii)).npulse(1),npulse));
       end
    end
    
    if isnan(sampler)
        zl = plsinfo('zl', plsgrp(1));
        sampler = awgdata.clk/(abs(zl(1, 1)) * max(1, awgdata.pulsegroups(plsgrp(1)).nrep(1)));       
        
        for ii=1:length(plsgrp)
           zl = plsinfo('zl',plsgrp(ii));
           sampler2 = awgdata.clk/(abs(zl(1, 1)) * max(1, awgdata.pulsegroups(plsgrp(ii)).nrep(1)));
           if(sampler ~= sampler2)
               warning(sprintf('Pulse length mismatch; sample rate for group %s is %g, not %g\n',...
                   awgdata.pulsegroups(plsgrp(ii)).name,sampler2,sampler));
           end
        end
    end    
    
    if (isfield(awgdata.pulsegroups, 'jump') && ~isempty(awgdata.pulsegroups(plsgrp(1)).jump)) || ...
            awgdata.pulsegroups(plsgrp(1)).nrep(1) == 0 || ...  % single pulse repeated indefinitely -> same logic                   
            awgdata.pulsegroups(plsgrp(1)).nrep(1) == 1  % single pulse repeated indefinitely -> same logic                   
        npulse = npulse*nloop;
        fastmode = 1;
    else
        nloop = 0;
    end
    
    seqind = [awgdata.pulsegroups(plsgrp).seqind];
    scan.data.pulsegroups = awgdata.pulsegroups(plsgrp);

else % single pulse, preceded by trigger. 
    % doesn't work like this
    seqind = awgseqind(abs(plsgrp))-1; % subtract pulse to jump to preceding trigp.
    if isnan(npulse)
        npulse=1; 
    end  
    
    if isnan(sampler)
        pd = awgdata.pulsedata(abs(plsgrp(1)));
        sampler = pd.clk/(pd.pulsetab(1, end) * pd.tbase);
    end
    
    if nloop > 0 % treat same way as jump sequence
        npulse = npulse*nloop;
        fastmode = 1;
    end
    % could allow mix of groups and fixed pulses
end


scan.configfn(1).fn = @smaconfigwrap;
scan.configfn(1).args = {smdata.inst(inst_index).cntrlfn [inst_index 0 99] [] [] config};
scan.configfn(2).fn = @smaconfigwrap;
scan.configfn(2).args = {smdata.inst(inst_index).cntrlfn,[inst_index 0 5]};

% arm card
scan.loops(1).prefn(1).fn = @smaconfigwrap;
scan.loops(1).prefn(1).args = {smdata.inst(inst_index).cntrlfn,[inst_index 0 4]};

% jump to correct pulseline
scan.loops(1).prefn(2).fn = @(x, seqind, loop) smset('PulseLine', seqind(mod(x(loop)-1, end)+1));
scan.loops(1).prefn(2).args = {seqind, 1};

scan.cleanupfn(1).fn =  @smaconfigwrap;
scan.cleanupfn(1).args = {@smset, {'PulseLine'}, 1};


scan.loops(1).getchan = [datachan auxchan];
scan.loops(1).setchan = [];
scan.loops(1).ramptime = [];
scan.loops(1).npoints = length(seqind);
scan.loops(1).rng = [];

if length(plsgrp) == 1 &&  dccomp == 0
    scan.loops(1).npoints = nrep;
    scan.loops(1).setchan = 'count';
else
    scan.loops(1).npoints = length(seqind);
    scan.loops(2).npoints = nrep;
    scan.loops(2).setchan = 'count';        
end

if strcmp(dispchan, 'default')
    dispchan = 1:numel(ops);
end

if genMask
	dispchan = 1:numel(operations);
	if numel(dispchan) > 4
		dispchan = dispchan(1:4);
	end
end

scan.disp=[];
for l=1:length(dispchan)    
% for l = 1:length(operations)
  scan.disp(end+1).loop = 1;
  scan.disp(end).channel = dispchan(l);
%   scan.disp(end).channel = l;
  scan.disp(end).dim = 1;

  scan.disp(end+1).loop = 1;
  scan.disp(end).channel = dispchan(l);
% 	scan.disp(end).channel = l;
  scan.disp(end).dim = 2;
end
    % scan.saveloop = [length(scan.loops), 50];

    % if npulse*oversamp/nloop == 1
    %     scan.disp(2) = []; % make disp work in degenerate case
    % end

    % switch fastmode
    %     case 1
    %         % remove inner loop for fast acq. 
    %         scan.configfn(2).args = {'fast', 1, [scan.loops(1).npoints, 0, 0, channelMask]};
    %         scan.loops(1) = [];
    %         scan.loops(1).prefn(1).args{2} = 1;
    %         scan.loops(end).setchan = 'count'; 
    %         scan.saveloop = [length(scan.loops) 50];
    %         [scan.disp.loop] = deal(1);
    % end
%% create more advanced polarization scan using polarizefn
% %%
if useFb
    % fbdata.x = plsinfo('xval',plsgrp(1));
    % fbdata.x=fbdata.x(fbdata.pulses);

    if(pulsetab(3) < 0)
      fbdata.fbx = plsinfo('xval',-pulsetab(3));
    end

    % add off pulse at end and return to measurement point
    pt = [pulsetab; 1 Inf, awgseqind(1), 0];

    np = size(pulsetab, 1);
    % pt(1:np, 3) = awgseqind(pulsetab(:, 3));     

    if offpos == -2;
        offpos = 2:np-1;
    end

    switch fbmode
        case 1 % software or manual  fb not yet done
            pt(1:2, 5) = [7 6];

        case 2 % fbdata.pulseind controlled
            pt(1:np, 5) = 3;       

        case 3 % pulseind is sequence line offset
            pt(1, 5) = 4;
            pt(end,:)=[];
            offpos = [];
    end

    for i = sort(offpos, 1, 'descend')
        pt = pt([1:i-1, end-1, i+1:end], :);
    end


    scan.data.polpulse =  awgdata.seqpulses(1:find(isfinite(awgdata.seqpulses), 1, 'last'));

    % loop whose index is taken to determined pulse
%     polloop = 3 - (length(plsgrp)==1 &&  dccomp==0) - (fastmode==1);
		polloop = 1;

    clear pf;
    pf.fn = @polarizefn;
    pf.args = {polloop, pt};
    scan.loops(1).postfn = pf;
%     scan.loops(2-(fastmode==1)).postfn(end+1) = pf;

if ~usePol
	scan.loops(end).datafn.fn = @gradfeedbfn;
	scan.loops(end).datafn.args = {};
end
end

scan.data.histrange = histRange;
scan.data.nloop = nloop;
end

%% utilities
function x = nloop_finder(T, npulse)

N = 100 * T * npulse;
x = lcm(256, N);
x = x / N;
end

%% down here live input validation functions
function valid = validate_plsgrp(plsgrp)
    if ischar(plsgrp)
        valid = ~isnan(awggrpind(plsgrp));
        return
    elseif iscellstr(plsgrp)
        valid = all(~isnan(awggrpind(plsgrp)));
    end
end

 function valid = validate_operations(operations)
    valid = true;
    if ~iscellstr(operations)
        valid = false;
        return
    end
    allowed = {'DS','RSA','HIST'};
    for op=operations
        valid = any(strcmp(op, allowed));
        if ~valid
            return
        end
    end
 end

 function valid = validate_dispchan(disp)
    valid = true;
    return
 end

%  function valid = validate_inline(inline)
%    if islogical(inline)
% 		 valid = true;
% 	 elseif strcmp(inline, 'block') || strcmp(inline, 'interleaved')
% 		 valid = true;
% 	 else
% 		 valid = false;
% 		 warning('inline must be true, false, ''block'' or ''interleaved''');
% 	 end
%  end