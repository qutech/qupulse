function transformation = to_transformation(mat_trafo)

trafo_module = py.importlib.import_module('qctoolkit._program.transformation');

if istable(mat_trafo)
    assert(size(mat_trafo, 1) == size(mat_trafo, 2));
    if isempty(mat_trafo.Properties.RowNames)
        mat_trafo.Properties.RowNames = mat_trafo.Properties.VariableNames;
    end
    
    
    kwargs = pyargs('columns', mat_trafo.Properties.VariableNames,...
                    'index', mat_trafo.Properties.RowNames');
    data = util.py.mat2py(mat_trafo{:,:});
    df = py.pandas.DataFrame(data, kwargs);
    
    transformation = trafo_module.LinearTransformation(df);
else
    error('invalid trafo type');
end