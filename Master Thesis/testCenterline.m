function testCenterline
if(exist('p','var')==0 || exist('weakLearners','var')==0 )
    fprintf('REQUIRED Parameters not present!!!\n')
else
newimg = rgb2gray(imread('A1.jpg'));
newimg_double= im2double(newimg);
normalizedImage = mat2gray(newimg_double);
%imshow(normalizedImage);
out_multiscale  = predictCenterlines(normalizedImage, weakLearners, wlMultiscale, p);
[out_score_max,out_scale_max ] = max(out_multiscale,[],p.dimension+1); % get maximum along scales 
nms_score =edgeNms(out_score_max,[],1,5); % apply non-maxima suppression to extract centerlines only
out_scale_max = out_scale_max + min_pred_scales - 1; % radius associated to the centerlines.
imshow(nms_score,[])
end
end

%tasks to do
% write super wrapper class to include :
% to run centerlines or boundary
% based of if the necessary parameters exist or not




% code to visualize nrrd data
%nrrdimg = permute(im2double(nrrdLoad('D:\Privat\pvjfiles\thesiswork\data\ny_roads\train\radial_gt\ny_2_radial_gt.nrrd')), [2 1 3 4]);
%imshow(img)

%%%problems::
%%% workspace variables not available when running the matlab probgram from
%%% toolbar. need to copy the code in command window so that the matlab
%%% program can access the variables.