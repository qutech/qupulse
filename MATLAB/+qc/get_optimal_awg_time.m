function [optimalTime, addTime, optimalNsamp] = get_optimal_awg_time(desiredTime, sampleRate)
	% Tabor AWG closest optimal number of samples: sampleQuantum + n*sampleQuantum
	% Time is in s, sampleRate is in Sa/s
	
	global plsdata
	
	sampleQuantum = plsdata.awg.sampleQuantum;
	minSamples = plsdata.awg.minSamples;
	
	global plsdata
	if nargin < 2 || isempty(sampleRate)
		sampleRate = plsdata.awg.sampleRate;
	end
	
	desiredNsamp = desiredTime*sampleRate;
	
	if desiredNsamp <= minSamples
		optimalNsamp = minSamples;
	else
		optimalNsamp = minSamples + sampleQuantum*round((desiredNsamp-minSamples)/sampleQuantum);
	end
	
	optimalTime = optimalNsamp / sampleRate;
	addTime = optimalTime - desiredTime;