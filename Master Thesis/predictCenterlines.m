
function score_img = predictCenterlines(input_img,weakLearners,wlMultiscale,p)
%predict centerlines of gray scale image input_img
%changing name input_img to img2

size_img = size(input_img(:,:,1));

n_train_scales = uint8(length(p.train_scales));
n_filter_banks = uint8(n_train_scales + uint8(p.n_ac_iter>0)); 

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

all_predict_scales_idx = cell(size(p.predict_scales_test));
all_predict_scales_idx{1} = 1:length(p.predict_scales_test{1});
for i_train_scale = 2:length(p.predict_scales_test),
    all_predict_scales_idx{i_train_scale} = [1:length(p.predict_scales_test{i_train_scale})] + length(all_predict_scales_idx{i_train_scale-1});
end



SEP_FEATURES = cell(1,n_train_scales); % large data
AC_FEATURES = cell(1,n_train_scales); % large data
%compute separable features image (if no enough memory, train one scale at a time)
for i_train_scale =1:n_train_scales,
   fprintf('computing separable features- train scale %i/%i  \n',...
       i_train_scale,n_train_scales);
    SEP_FEATURES{i_train_scale} = convolve_separable(input_img,...
        separable_filters{i_train_scale},p.predict_scales_test{i_train_scale}./p.train_scales(i_train_scale)); % compute features also with rescaled filters
end

score_img = zeros([size_img,numel(cell2mat(all_predict_scales_idx))],'single');
for i_ac = 0:p.n_ac_iter,
     fprintf('AC iter %d/%d\n',i_ac,p.n_ac_iter);
     for i_train_scale = 1:n_train_scales, 
        n_pred_scales_i_train = length(p.predict_scales_test{i_train_scale});
        for j_pred_scale = 1:n_pred_scales_i_train,
            scale_factor = p.predict_scales_test{i_train_scale}(j_pred_scale)/p.train_scales(i_train_scale);
            fprintf('Training scale: %d/%d\n',i_train_scale,n_train_scales);
            fprintf('Predicting scale %d/%d\n',j_pred_scale,n_pred_scales_i_train);
             [  score_img(:,:,all_predict_scales_idx{i_train_scale}(j_pred_scale) )]=...
                        applyRegTreeMex_centerlines_par(weakLearners{i_ac+1,i_train_scale},...
                        SEP_FEATURES{i_train_scale}(:,:,j_pred_scale),AC_FEATURES{i_train_scale}(:,:,j_pred_scale*(i_ac>0)+1*(i_ac==0)),...
                        size_img,weights,i_train_scale,scale_factor,i_ac,p.n_precompute_features_image,p.n_precompute_features_ac);
        end

     end
    if(p.n_ac_iter>0 && i_ac<p.n_ac_iter) 

        for i_train_scale =1:n_train_scales,
            n_pred_scales_i_train = length(p.predict_scales_test{i_train_scale});
            AC_FEATURES{i_train_scale} = zeros([prod(size_img),size(weights{end},1),n_pred_scales_i_train],'single'); 
            for j_pred_scale = 1:n_pred_scales_i_train,
                fprintf('computing separable features test score - train scale %d/%d - scale %i/%i  \n',...
                            i_train_scale,n_train_scales,j_pred_scale,length(p.predict_scales_test{i_train_scale}));
                    AC_FEATURES{i_train_scale}(:,:,j_pred_scale) = convolve_separable(score_img(:,:,all_predict_scales_idx{i_train_scale}(j_pred_scale)),...
                              separable_filters{end});
            end
        end             
     end % end compute separable features ac
end% end ac iters


if(p.multiscale_ac && numel(cell2mat(all_predict_scales_idx)) >1)
    [score_img] = predict_MC_given_score(wlMultiscale,score_img,p.pooling_steps_MC,p.pooling_win_size_MC);
    
    
end 
end

