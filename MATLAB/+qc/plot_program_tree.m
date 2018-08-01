function plot_program_tree(program, maxElements, figId)
	% Modified from Wolfie at
	% https://stackoverflow.com/questions/45666560/ploting-a-nested-cell-as-a-tree-using-treeplot-matlab/45676012

	if nargin < 2 || isempty(maxElements)
		maxElements = 10;
	end
	if nargin < 3 || isempty(figId)
		figId = 120;
	end
		
	[treearray, nodeVals] = getTreeArray(program, maxElements);
	
	figure(figId); clf;
	treeplot(treearray);
	title(sprintf(['Duration %g' 10 'Showing first %g entries'], double(program.duration.numerator/program.duration.denominator), maxElements))
	
	% Get the position of each node on the plot
	[x, y] = treelayout(treearray);
	
	% Get the indices of the nodes which have values stored
	nodeIndices = cell2mat(nodeVals(1,:));
	
	% Get the labels (values) corresponding to those nodes. Must be strings in cell array
	labels = cellfun(@(x)(double(x.repetition_count)), nodeVals(2,:), 'uniformoutput', 0);
	
	% Add labels, with a vertical offset to the y coords so that labels don't sit on nodes
	text(x(nodeIndices), y(nodeIndices) - 0.03, labels);
	
end

function [treearray, nodeVals] = getTreeArray(program, maxElements)
	% Initialise the array construction from node 0
	
	children = program.children(1:min(maxElements, end));
	
	[nodes, ~, nodeVals] = treebuilder(children, 1);
	treearray = [0, nodes];
	nodeVals(:, end+1) = {1; program};
	
	% Recursive tree building function
	function [nodes, currentNode, nodeVals] = treebuilder(children, rootNode)
		% Set up variables to be populated whilst looping
		nodes = []; nodeVals = {};
		
		% Start node off at root node
		currentNode = rootNode;
		
		% Loop over array elements, either recurse or add node
		for ii = 1:min(size(children, 2))
			currentNode = currentNode + 1;
			try
				nodeVals = [nodeVals, {currentNode; children{ii}}];
				[subtreeNodes, currentNode, newNodeVals] = treebuilder(children{ii}.children, currentNode);
				nodes = [nodes, rootNode, subtreeNodes];
				nodeVals = [nodeVals, newNodeVals];
			end
		end
	end
end