
set_global_variable('workspace_path', fileparts(mfilename('fullpath')));

set_global_variable('version', 6);

% Enable more verbose output
% set_global_variable('debug', 1);

% Disable result caching
% set_global_variable('cache', 0);

% Select experiment stack
set_global_variable('stack', 'votlt2018');
set_global_variable('python', 'env -i /home/space/Public/anaconda3/envs/iccv2019/bin/python');
