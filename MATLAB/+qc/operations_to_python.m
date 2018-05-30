function pyOperations = operations_to_python(operations)
	% Convert operations struct from matlab to python
		
	pyOperations = {};	
	
	for k = 1:numel(operations)
		
		if numel(operations{k}) > 1
			args = operations{k}(2:end);
		else
			args = {};
		end
		switch(operations{k}{1})
			case 'AlgebraicMoment'
				pyOp = py.atsaverage.operations.AlgebraicMoment(args{:});
			case 'Downsample'
				pyOp = py.atsaverage.operations.Downsample(args{:});
			case 'Histogram'
				pyOp = py.atsaverage.operations.Histogram(args{:});
			case 'RepAverage'
				pyOp = py.atsaverage.operations.RepAverage(args{:});
			case 'RepeatedDownsample'
				pyOp = py.atsaverage.operations.RepeatedDownsample(args{:});
			otherwise
			  error('Operation %s not recognized', operations{k}{1});
		end		
		pyOperations{end+1} = pyOp;	
		
	end
	
	pyOperations = py.list(pyOperations);


 
