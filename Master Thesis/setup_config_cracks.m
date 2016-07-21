function p = setup_config_cracks()
% configuration parameters for cracks dataset 
% used by runCenterlinesTrainTestCrack

%% dataset parameters

p.dimension = uint8(2); % dimension of images 
p.dataset_name = 'cracks'; % dataset name, used to find images lists
p.tr = 0;
p.color_img = 0;

%train lists 
p.train_img_list_filename = fullfile('lists',p.dataset_name,'crack_imgs.txt');
p.train_radial_gt_list_filename = fullfile('lists',p.dataset_name,'crack_gt_radial.txt');

%test lists
p.test_img_list_filename = fullfile('lists',p.dataset_name,'test_imgs.txt');
p.test_radial_gt_list_filename = fullfile('lists',p.dataset_name,'test_gt_radial.txt');


%% filters parameters

%2 TRAIN SCALES + auto-context (AC) AC = auto context
 %original filters
 p.learned_filters_path ={'data/cracks/filter_banks/roads_121_rec_cpd_rank_49.txt';... % filters used on 1st train scale
     'data/cracks/filter_banks/roads_121_rec_cpd_rank_49.txt';... % filters used on 2nd train scale
 'data/cracks/filter_banks/ny_roads_dist_gt_21_36_rec_cpd_rank_16.txt'}; % filters used on auto-context scores
%separable filters 
p.separable_filters_path ={'data/cracks/filter_banks/roads_121_sep_cpd_rank_49.txt';...
     'data/cracks/filter_banks/roads_121_sep_cpd_rank_49.txt';...
 'data/cracks/filter_banks/ny_roads_dist_gt_21_36_sep_cpd_rank_16.txt'};
%weights
 p.weights_path ={'data/cracks/filter_banks/roads_121_weigths_cpd_rank_49.txt';...
     'data/cracks/filter_banks/roads_121_weigths_cpd_rank_49.txt';...
 'data/cracks/filter_banks/ny_roads_dist_gt_21_36_weigths_cpd_rank_16.txt'};

p.filters_size = {uint8([21,21]);uint8([21,21]);uint8([21,21])};

p.select_filters = {false;false;false}; % if true, select only a subset of the filters to compute the features (accordingly to p.select_idx)
p.select_filters_idx = {[];[];[]}; % indexed of filters to select to compute features (used only if p.select_filters = true)

%% context features parameters
p.n_feat_center_img = uint32(50); % features on image computed at center
p.n_feat_cont_img = uint32(500); % features on image using context
p.n_feat_center_ac = uint32(30); % features on score computed at center
p.n_feat_cont_ac = uint32(500); % features on score usign context
p.max_cont_step = int8(13); % max length of ray used for context features 

%precompute non-separable features for testing
p.n_precompute_features_image = 121;
p.n_precompute_features_ac = 121;

%% multiscale parameters
p.all_scales = single([5:10]); % total number of scales predicted (also used to compute gt) % original line
p.scale_toll = single(0.5*ones(size(p.all_scales))); % tolerance for discretizing scale domain

% 2 TRAIN SCALES
p.train_scales = single([6 9]); % scales used to train regressors %original

%p.predict_scales = {single([5:7]),single([8:10])}; % p.predict_scales{j} are scales predicted using regressor trained at scale p.train_scales(j) 
p.predict_scales = {single([6]),single([9])}; %p.predict_scales{j} are scales predicted using regressor trained at scale p.train_scales(j)%original

p.predict_scales_test=p.predict_scales; %these should always be the same !
p.sample_scales= p.predict_scales; %


%% gradient boost parameters
%training samples
%p.neg_sample_no = uint32(2000*ones(1,length(p.train_scales))); %number of samples far from centerlines % original line
p.neg_sample_no = uint32(8000*ones(1,length(p.train_scales))); %number of samples far from centerlines
p.pos_sample_no = uint32(8000*ones(1,length(p.train_scales))); %number of samples close to centerlines % original value 2000*ones...
%p.T2Size = uint32(1000); % number of samples used at each boosting iteration %original
p.T2Size = uint32(6000);

p.iters_no = uint16(100); % number of weak learners ( i.e. number of boosting iterations ) % original value was 25
p.tree_depth_max = uint8(2); % max deepth of a tree ( each tree is a weak learner ) % original value was 1  
p.shrinkage_factor = single(0.1); % shrinkage factor multiplied at score at each iteration 
p.loss_type = 'squared'; % loss to minimize at each iteration (on sampled pixels)

%% autocontext parameters
p.n_ac_iter = uint8(2); % number of autoncontext iterations % original value was 1, changed to 2!  

%% saving parameters
p.codename = sprintf('%s_loss_%s_pos_%d_neg_%d_T2_%d_tree_depth_%d_boost_iters_%d_ac_iters_%d', ...
    p.dataset_name,p.loss_type,p.pos_sample_no(1),p.neg_sample_no(1),p.T2Size, ...
    p.tree_depth_max,p.iters_no,p.n_ac_iter);
p.results_dir = fullfile('results',p.dataset_name,p.codename);

%% post processing parameters
p.do_nms = true; % do non-maxima suppression on images

%% multithread parameters
 p.omp_num_threads = '12';
 p.parfor_num_threads = 8;
 p.predict_parallel = 0;
 p.convolve_parallel = 0;
 

 %% multiscale last step parameters
 p.multiscale_ac  = 0; % original value was 1   
 
%train parameters
p.n_iters_MC = 100; % original value was 25  
p.n_pos_MC = 1000000; % original value was 10000  
p.n_neg_MC = 1000000; % original value was 10000  
p.toll_pos_MC = 10;

%optimization params
p.opts_MC.loss = 'squaredloss'; 
p.opts_MC.shrinkageFactor = 0.1;
p.opts_MC.subsamplingFactor = 0.5;
p.opts_MC.maxTreeDepth= uint32(2);
p.opts_MC.disableLineSearch = uint32(0);
p.opts_MC.mtry = uint32(300);

p.pooling_win_size_MC = [0 0 0 1 3 5 5 5 7 11]; % must be sorted
p.pooling_steps_MC = [0 1 2 0 4 0 4 8 0 4];

p.debug = 1;% original value was 1 
 