% Created with help of :
% runCenterlinesTrainTest.m 
% - train and test regression-based algorithm to predict locations and
% radius of cracks
%  centerlines in images
% - learn a scaled and thresholded distance transform to the
%     centerline as in [1]
% - a cascade of regressors is used to improve the results
% - features are extracted from the image via convolutions with a given
% bank of filters.
%     The filters must be linear combination of separable ones
% see [2] to know how to obtain such filter banks, or use the ones provided
% with this code.
%
%
% [1] A. Sironi, V. Lepetit and P. Fua. 'Multiscale Centerline Detection by Learning a Scale-Space Distance Transform" CVPR 2014.
% [2] A. Sironi, B. Tekin, R. Rigamonti, V. Lepetit and P. Fua. 'Learning Separable Filters'. PAMI 2014.
%
%//////////////////////////////////////////////////////////////////////////////////
%//																				 //																			 //
%// Copyright (C) 2014 Amos Sironi												 //
%//																				 //																			 //
%// This program is free software: you can redistribute it and/or modify         //
%// it under the terms of the version 3 of the GNU General Public License        //
%// as published by the Free Software Foundation.                                //
%//                                                                              //
%// This program is distributed in the hope that it will be useful, but          //
%// WITHOUT ANY WARRANTY; without even the implied warranty of                   //
%// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU             //
%// General Public License for more details.                                     //
%//                                                                              //
%// You should have received a copy of the GNU General Public License            //
%// along with this program. If not, see <http://www.gnu.org/licenses/>.         //
%//                                                                              //
%// Contact <amos.sironi@epfl.ch> for comments & bug reports                     //
%//////////////////////////////////////////////////////////////////////////////////
%
%

clearvars;

addpath(genpath('sampling'));
addpath(genpath('util'));
addpath(genpath('evaluation'));
addpath(genpath('mex')); % put here compiled mex files (see README.txt for more details)
addpath(genpath('helper_functions'));
addpath(genpath('util'))
%addpath(genpath('util/MinMaxFilterFolder'))% 
addpath(genpath('/cvlabdata1/home/asironi/MATLAB/tensor_toolbox')); 
addpath(genpath('/cvlabdata1/home/asironi/MATLAB/contour_detection/dollar_toolbox'))
addpath(genpath('D:/Privat/pvjfiles/thesiswork/contrib/tensor')); % uncomment and set path to tensor toolbox (http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.5.html)
addpath(genpath('D:/Privat/pvjfiles/thesiswork/contrib/dollar'))% uncomment and set path to Piotr Dollar toolbox (http://vision.ucsd.edu/~pdollar/toolbox/doc/)

%for params tuning
rng(1234);


%% get configuration =====================================================
fprintf('Setting up configuration\n');
[p] = setup_config_cracks(); % load parameters 

[p] = setup_directories(p); % create results directories


setenv('OMP_NUM_THREADS', p.omp_num_threads);  

% end get config =========================================================

%% load data =============================================================

fprintf('Loading train images...\n');
[train_imgs,train_radial_gts] = load_dataset(p,'train');


if(p.debug)
   rand_imgs = randperm(length(train_imgs),5); % for debug use only 5 image
   train_imgs = train_imgs(rand_imgs);  
   train_radial_gts = train_radial_gts(rand_imgs);
end


n_train_images = length(train_imgs);

%get size images
size_train_images = cell(1,n_train_images);
for i_img = 1:n_train_images,
    size_train_images{i_img} = (size(train_imgs{i_img})); 
end
% end load data ----------------------------------------------------------

%% compute distance gt -----------------------------------------------------------
fprintf('Computing distance gt...\n');
    
train_dist_gts = cell(n_train_images,1); % 
for i_img = 1:n_train_images,    
    fprintf('...img %d/%d...\n',i_img,n_train_images);
    train_dist_gts{i_img} = compute_exp_distance_transform(train_radial_gts{i_img},p.all_scales,p.scale_toll);
end
% end compute gt -------------------------------------------------------------

% load filters -----------------------------------------------------------

%one filter bank for each train scale and one filter bank for ac iter
%check no of filters:
n_train_scales = uint8(length(p.train_scales));
n_filter_banks = uint8(n_train_scales + uint8(p.n_ac_iter>0)); 
if(length(p.separable_filters_path)<n_filter_banks)
    error('incorrect number of filter banks: given %i, needed %i\n',length(p.separable_filters_path),n_filter_banks)
end

weights = cell(1,n_filter_banks);
separable_filters = cell(1,n_filter_banks);
filters_no = zeros(1,n_filter_banks,'uint32'); % number of full rank filter banks
for i_fb = 1:n_filter_banks
    weights{i_fb} = single(load(p.weights_path{i_fb})); 
    [separable_filters{i_fb}] = load_separable_filter_bank(p.separable_filters_path{i_fb});

    %select subset of filters 
    if(p.select_filters{i_fb})
        weights{i_fb} = weights{i_fb}(:,p.select_filters_idx{i_fb});
    end
    filters_no(i_fb) = size( weights{i_fb},2);    
    
end

% end load filters -------------------------------------------------------

%if use parallel
%matlabpool(p.parfor_num_threads)
if( p.convolve_parallel)
 pools = matlabpool('size');
 cpus = feature('numCores');
 if pools ~= (p.parfor_num_threads)
     if pools > 0
         matlabpool('close');
     end
     if(cpus > p.parfor_num_threads)
         matlabpool('open', p.parfor_num_threads);
     else
         matlabpool('open', cpus);
     end
 end
end


% compute all sep features at the beginning --------
SEP_FEATURES = cell(n_train_images,n_train_scales);
for i_img = 1:n_train_images,
   for i_train_scale =1:n_train_scales,
       fprintf('computing separable features train img %i/%i - train scale %i/%i  \n',...
           i_img,n_train_images,i_train_scale,n_train_scales);
        SEP_FEATURES{i_img,i_train_scale} = convolve_separable(train_imgs{i_img},...
            separable_filters{i_train_scale},p.predict_scales{i_train_scale}./p.train_scales(i_train_scale)); % compute features also with rescaled filters
   end
end

clear filters_full_rank
% end precompute features ------------------------------------------------

% ===========================================================

%% START TRAINING ========================================================
tic
if(p.predict_parallel )
 pools = matlabpool('size');
 cpus = feature('numCores');
 if pools ~= (p.parfor_num_threads)
     if pools > 0
         matlabpool('close');
     end
     if(cpus > p.parfor_num_threads)
         matlabpool('open', p.parfor_num_threads);
     else
         matlabpool('open', cpus);
     end
 end
end

%get scales
all_predict_scales_idx = cell(size(p.predict_scales));
all_predict_scales_idx{1} = 1:length(p.predict_scales{1});
for i_train_scale = 2:length(p.train_scales),
    all_predict_scales_idx{i_train_scale} = [1:length(p.predict_scales{i_train_scale})] + length(all_predict_scales_idx{i_train_scale-1});
end

%initialize
weakLearners = cell(p.n_ac_iter+1,n_train_scales);
AC_FEATURES = cell(n_train_images,n_train_scales);
score_imgs = [];

% start autocontext iterations
for i_ac = 0:p.n_ac_iter,
    fprintf('AC iter: %i\n', i_ac);
    for i_train_scale = 1:n_train_scales,
        fprintf('Training scale: %1.2f\n', p.train_scales(i_train_scale));
        
        %get samples
        fprintf('Getting samples from training images\n');    
        SAMPLES = get_samples_cont(p,train_imgs,train_dist_gts,train_radial_gts,i_train_scale);

        % train now !
        fprintf('Starting boosting iterations\n');        
        [weakLearners{i_ac+1, i_train_scale}] = trainBoostGeneral_cont(SAMPLES,p,SEP_FEATURES(:,i_train_scale),weights,size_train_images,AC_FEATURES(:,i_train_scale),filters_no,i_train_scale,i_ac);
        % in the above line boosting iterations occur!
    end % end train scales


    % save temporary regressor
    fprintf('The regressors at AC iter %i have been trained, storing them...\n',i_ac);
    nome_wl = sprintf('wl_final_scales_%1.2f_%1.2f_temp.mat',p.predict_scales{1}(1),p.predict_scales{end}(end));
    save(fullfile(p.results_dir,nome_wl),'weakLearners', 'p','-v7.3');

    % predict score images to compute features and to train last regressor
        score_imgs = cell(1,n_train_images);
        for i_img = 1:n_train_images,
            score_imgs{i_img} = zeros([size_train_images{i_img},1,numel(cell2mat(all_predict_scales_idx))],'single');
            for i_train_scale = 1:n_train_scales,
                n_pred_scales_i_train = length(p.predict_scales{i_train_scale});
                % predict scales for i_train_scale (with context) (only for ac )
                for j_pred_scale = 1:n_pred_scales_i_train,
                    scale_factor = p.predict_scales{i_train_scale}(j_pred_scale)/p.train_scales(i_train_scale);
                    fprintf('AC iter: %i, Training scale: %1.2f, Image: %i/%i\n',i_ac,p.train_scales(i_train_scale),i_img,n_train_images);
                    fprintf('Predicting scale %d/%d\n',j_pred_scale,n_pred_scales_i_train);
                    % predict scale j_pred_scale on train images
                        score_imgs{i_img}(:,:,1,all_predict_scales_idx{i_train_scale}(j_pred_scale) ) =...
                            applyRegTreeMex_centerlines_par(weakLearners{i_ac+1,i_train_scale},...
                                    SEP_FEATURES{i_img,i_train_scale}(:,:,j_pred_scale),AC_FEATURES{i_img,i_train_scale}(:,:,j_pred_scale*(i_ac>0)+1*(i_ac==0)),...
                                    size_train_images{i_img},weights,i_train_scale,scale_factor,i_ac,p.n_precompute_features_image,p.n_precompute_features_ac);
                end
            end
        end
    
     
     % compute features on score image of previous ac iteration
     if(p.n_ac_iter>0 && i_ac < p.n_ac_iter )     
        for i_img = 1:n_train_images,
            for i_train_scale =1:n_train_scales,
                n_pred_scales_i_train = length(p.predict_scales{i_train_scale});
                AC_FEATURES{i_img,i_train_scale} = zeros([prod(size_train_images{i_img}),size(weights{end},1),n_pred_scales_i_train],'single'); 
                for j_pred_scale = 1:n_pred_scales_i_train,
                    
                    fprintf('computing separable features train score %i/%i - train scale %d/%d - scale %i/%i  \n',...
                                i_img,n_train_images,i_train_scale,n_train_scales,j_pred_scale,length(p.predict_scales{i_train_scale}));
                        AC_FEATURES{i_img,i_train_scale}(:,:,j_pred_scale) = convolve_separable(score_imgs{i_img}(:,:,1,all_predict_scales_idx{i_train_scale}(j_pred_scale)),...
                                  separable_filters{end});
                end
            end
        end
        clear train_imgs_temp

     end % end compute separable features ac
    
end % end autocontext iter

% last step: combine output of all scales
wlMultiscale = [];
if(p.multiscale_ac && numel(cell2mat(all_predict_scales_idx)) >1)% not run/relevant if multiscale=0
    % combine score images in last step
    [wlMultiscale] = learn_MC_given_score(score_imgs,train_dist_gts,p.n_pos_MC,p.n_neg_MC,p.toll_pos_MC,p.n_iters_MC,p.opts_MC,p.pooling_steps_MC,p.pooling_win_size_MC);
end
nome_wl_temp = fullfile(p.results_dir,nome_wl);

% save results
fprintf('The classifier has been trained, storing it...\n');
nome_wl = sprintf('wl_final_scales_%1.2f_%1.2f.mat',p.predict_scales{1}(1),p.predict_scales{end}(end));
save(fullfile(p.results_dir,nome_wl),'wlMultiscale','weakLearners', 'p','-v7.3');

system(sprintf('rm -f %s',nome_wl_temp));

training_time_250iter=toc;
%free memory
clear SAMPLES SEP_FEATURES AC_FEATURES score_imgs train_imgs train_dist_gts train_radial_gts train_masks train_neg_masks

%% START (internal) TESTING


%% load data =============================================================

tic
% load test images, masks and radial gt ---------------------------------------
fprintf('Loading test images...\n');
[test_imgs,test_radial_gts] = load_dataset(p,'test');
n_test_images = length(test_imgs);
n_test_images = uint8(n_test_images);


%get size images
size_test_images = cell(1,n_test_images);
for i_img = 1:n_test_images,
    size_test_images{i_img} = double(size(test_imgs{i_img}));
end

% end load test images ---------------------------------------------------

if(p.do_nms)
    mkdir([p.test_subdir_path,'/nms_images'])
end

%% predict test images ===================================================
fprintf('Predicting test images...\n');

score_imgs = cell(1,n_test_images);
[test_img_list,test_imgs_no] = get_list(p.test_img_list_filename);

min_pred_scales = min(min(cell2mat(p.predict_scales_test))); % 

all_predict_scales_idx = cell(size(p.predict_scales_test));
all_predict_scales_idx{1} = 1:length(p.predict_scales_test{1});
for i_train_scale = 2:length(p.predict_scales_test),
    all_predict_scales_idx{i_train_scale} = [1:length(p.predict_scales_test{i_train_scale})] + length(all_predict_scales_idx{i_train_scale-1});
end

% 

if(p.debug) % for debug test on 1 image
    n_test_images = 1;
end


for i_img = 1:n_test_images,
    fprintf('Predicting img %d/%d\n',i_img,n_test_images);
    
    
    SEP_FEATURES = cell(1,n_train_scales); % 
    AC_FEATURES = cell(1,n_train_scales); % 
    for i_train_scale =1:n_train_scales,
       fprintf('computing separable features- train scale %i/%i  \n',...
           i_train_scale,n_train_scales);
        SEP_FEATURES{i_train_scale} = convolve_separable(test_imgs{i_img},...
            separable_filters{i_train_scale},p.predict_scales_test{i_train_scale}./p.train_scales(i_train_scale)); % compute features also with rescaled filters
    end
    
    score_imgs{i_img} = zeros([size_test_images{i_img},numel(cell2mat(all_predict_scales_idx))],'single');
    for i_ac = 0:p.n_ac_iter,
         fprintf('AC iter %d/%d\n',i_ac,p.n_ac_iter);
         for i_train_scale = 1:n_train_scales, 
            n_pred_scales_i_train = length(p.predict_scales_test{i_train_scale});
            for j_pred_scale = 1:n_pred_scales_i_train,
                scale_factor = p.predict_scales_test{i_train_scale}(j_pred_scale)/p.train_scales(i_train_scale);
                fprintf('Training scale: %d/%d\n',i_train_scale,n_train_scales);
                fprintf('Predicting scale %d/%d\n',j_pred_scale,n_pred_scales_i_train);
                % predict scale j_pred_scale on test images
                      [  score_imgs{i_img}(:,:,all_predict_scales_idx{i_train_scale}(j_pred_scale) )]=...
                            applyRegTreeMex_centerlines_par(weakLearners{i_ac+1,i_train_scale},...
                            SEP_FEATURES{i_train_scale}(:,:,j_pred_scale),AC_FEATURES{i_train_scale}(:,:,j_pred_scale*(i_ac>0)+1*(i_ac==0)),...
                            size_test_images{i_img},weights,i_train_scale,scale_factor,i_ac,p.n_precompute_features_image,p.n_precompute_features_ac);
            end
         
         end

         %compute separable features score
         if(p.n_ac_iter>0 && i_ac<p.n_ac_iter) % not needed at last ac iter


            for i_train_scale =1:n_train_scales,
                n_pred_scales_i_train = length(p.predict_scales_test{i_train_scale});
                AC_FEATURES{i_train_scale} = zeros([prod(size_test_images{i_img}),size(weights{end},1),n_pred_scales_i_train],'single'); 
                for j_pred_scale = 1:n_pred_scales_i_train,
                    fprintf('computing separable features test score - train scale %d/%d - scale %i/%i  \n',...
                                i_train_scale,n_train_scales,j_pred_scale,length(p.predict_scales_test{i_train_scale}));
                        AC_FEATURES{i_train_scale}(:,:,j_pred_scale) = convolve_separable(score_imgs{i_img}(:,:,all_predict_scales_idx{i_train_scale}(j_pred_scale)),...
                                  separable_filters{end});
                end
            end             
         end % end compute separable features ac        
    end% end ac iters
    
    
    if(p.multiscale_ac && numel(cell2mat(all_predict_scales_idx)) >1)
        [score_imgs{i_img}] = predict_MC_given_score(wlMultiscale,score_imgs{i_img},p.pooling_steps_MC,p.pooling_win_size_MC);%does not execute when multiscale=0
 
    end
    
    % save score
      [~,name_im,~] = fileparts(test_img_list{i_img});
      name_output_scale_space = strcat(name_im,'_scale_space_tubularity');
      test_score_filename_multiscale = fullfile(p.test_subdir_path,sprintf('%s.nrrd',name_output_scale_space));  
      nrrdSave(test_score_filename_multiscale,permute(score_imgs{i_img},[2,1,3,4]));

    
    
    % non maximum suppression
    if(p.do_nms)
        [score_max,score_max_scale ] = max(score_imgs{i_img},[],p.dimension+1);
        name_output_tubularity = strcat(name_im,'_tubularity');
        test_score_filename_max_multiscale = fullfile( p.test_subdir_path,sprintf('%s.nrrd',name_output_tubularity));         
        nrrdSave(test_score_filename_max_multiscale,permute(score_max,[2,1,3,4]));
        name_output_scale = strcat(name_im,'_scale');
        test_score_filename_scale_multiscale = fullfile( p.test_subdir_path,sprintf('%s.nrrd',name_output_scale));  
        score_max_scale = score_max_scale + min_pred_scales - 1;
        nrrdSave(test_score_filename_scale_multiscale,permute(score_max_scale,[2,1,3,4]));
        
        name_output_supp = strcat(name_im,'_tubularity_supp');
        test_score_filename_nms_multiscale = fullfile( p.test_subdir_path,'nms_images',sprintf('%s.nrrd',name_output_supp));                
            

        nms_score =edgeNms(score_max,[],1,5);
        nrrdSave(test_score_filename_nms_multiscale,permute(nms_score,[2,1,3,4]));
   

    end % end nms
end % end test images
testing_time_250iter=toc;

if(p.predict_parallel || p.convolve_parallel)
    matlabpool close
end

fprintf('Program terminated normally.\n');

% end everithing


