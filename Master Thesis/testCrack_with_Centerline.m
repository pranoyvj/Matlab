function testCrack=testCrack_with_Centerline(weakLearners,wlMultiscale,p,s)
% count=1;
% count2=7;
% count3=15;
% 
% 
% for a = 44:47
%     filename = ['DSC016' num2str(a,'%2d') '.jpg'];    
%     tic
%     newimg = rgb2gray(imread(filename));
%     newimg_double= im2double(newimg);
%     normalizedImage = mat2gray(newimg_double); % normalize image between 0 and 1 value
%     out_multiscale  = predictCenterlines(normalizedImage, weakLearners, wlMultiscale, p);
%     [out_score_max,out_scale_max ] = max(out_multiscale,[],p.dimension+1); % get maximum along scales 
%     %nms_score =edgeNms(out_score_max,[],1,5); % apply non-maxima suppression to extract centerlines only %original line
%     nms_score =edgeNms(out_score_max,[],125,5);
%     out_scale_max = out_scale_max + min_pred_scales - 1; % radius associated to the centerlines.
%     time2(count2)=toc;
%     count2= count2+1;
%     figure(a*1);
%     imshow(nms_score,[]);
%    
% end
% for a = 1:3
%     filename = ['B' num2str(a,'%2d') '.jpg'];   
%     tic
%     newimg = rgb2gray(imread(filename));
%     newimg_double= im2double(newimg);
%     normalizedImage = mat2gray(newimg_double);
%     out_multiscale  = predictCenterlines(normalizedImage, weakLearners, wlMultiscale, p);
%     [out_score_max,out_scale_max ] = max(out_multiscale,[],p.dimension+1); % get maximum along scales 
%     %nms_score =edgeNms(out_score_max,[],1,5); % apply non-maxima suppression to extract centerlines only %original line
%     nms_score =edgeNms(out_score_max,[],125,5);
%     out_scale_max = out_scale_max + min_pred_scales - 1; % radius associated to the centerlines.
%     time(count)=toc;
%     count= count+1;
%     figure(a*1);
%     imshow(nms_score,[]);
%    
% end
% for a = 7:10
%     filename = ['B' num2str(a,'%2d') '.jpg'];   
%     tic
%     newimg = rgb2gray(imread(filename));
%     newimg_double= im2double(newimg);
%     normalizedImage = mat2gray(newimg_double);
%     out_multiscale  = predictCenterlines(normalizedImage, weakLearners, wlMultiscale, p);
%     [out_score_max,out_scale_max ] = max(out_multiscale,[],p.dimension+1); % get maximum along scales 
%     %nms_score =edgeNms(out_score_max,[],1,5); % apply non-maxima suppression to extract centerlines only %original line
%     nms_score =edgeNms(out_score_max,[],125,5);
%     out_scale_max = out_scale_max + min_pred_scales - 1; % radius associated to the centerlines.
%     time2(count2)=toc;
%     count2= count2+1;
%     figure(a*1);
%     imshow(nms_score,[]);
%    
% end
    tic
    %wins(1)=2*ceil(3*1)+1;% pre processing fujita
    newimg = rgb2gray(imread('D:\Privat\pvjfiles\thesiswork\data\cracks\testingcases\easycase\DSC01633.JPG'));
    %subtr = imsubtract(medfilt2(newimg, [20 20]), newimg);% pre processing fujita subtraction
    %gaussian = imfilter(subtr, fspecial('gaussian', wins(1), 1));% pre processing  smoothing
    %J = im2bw(gaussian, 0.1);                                      %pre processing thresholding
    newimg_double= im2double(newimg);
    normalizedImage = mat2gray(newimg_double);
    out_multiscale  = predictCenterlines(normalizedImage, weakLearners, wlMultiscale, p);
    [out_score_max,out_scale_max ] = max(out_multiscale,[],p.dimension+1); % get maximum along scales 
    %nms_score =edgeNms(out_score_max,[],1,5); % apply non-maxima suppression to extract centerlines only %original line
    nms_score =edgeNms(out_score_max,[],125,5);
    out_scale_max = out_scale_max + min_pred_scales - 1; % radius associated to the centerlines.
    toc
    figure();
    imshow(nms_score,[]);
    imwrite(nms_score, 'DSC01633_crack_100iter.jpg');
  
