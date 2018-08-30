function transformation = to_transformation(mat_trafo)
	
	trafo_module = py.importlib.import_module('qctoolkit._program.transformation');
	
	if istable(mat_trafo)
		assert(size(mat_trafo, 1) == size(mat_trafo, 2));
		if isempty(mat_trafo.Properties.RowNames)
			mat_trafo.Properties.RowNames = mat_trafo.Properties.VariableNames;
		end
		
		data = util.py.mat2py(mat_trafo{:,:});
		
		transformation = trafo_module.LinearTransformation(data, mat_trafo.Properties.RowNames', mat_trafo.Properties.VariableNames);
	elseif isempty(mat_trafo)
		transformation = py.None;
	else
		error('invalid trafo type');
	end