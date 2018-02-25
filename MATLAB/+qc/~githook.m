function scan = githook(scan, paths)
    root = git('rev-parse --show-toplevel');
		root = root(1:end-1); % remove \n
% 		dir = pwd;
% 		cd(root)
    paths = fullfile(root, paths);
		for i=1:numel(paths)
			paths{i} = ['"' paths{i} '"'];
		end
    git('add', paths{:});
		staged = git('diff' ,'--cached');
		if numel(staged) > 0
			git('commit -m "automated commit"');
		end
    scan.data.gitrepo = root;
    scan.data.gitrev = git('rev-parse HEAD');
% 		cd(dir);
end
