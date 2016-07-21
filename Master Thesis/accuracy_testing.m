function accuracy_testing
fprintf('\nstart----------------------------------------------------------\n');
fprintf('First , calculation with crack-centerline output image : \n');
img= imread('D:\Privat\pvjfiles\thesiswork\results\cracks\Results of easy case\Result with other boosting iterations\copy\DSC01633_crack_200iter_2208.48seconds.jpg');
img = rgb2gray(img);
[y,x] = size(img);
I_double = im2double(img);
blackpixels_index = find(I_double==0);
fprintf('The number of black pixels in this image is :'); % non black pixels contain data...so they are of interest!
bp= size(blackpixels_index);
disp(bp(1,1));
nbp=y*x-bp(1,1);
fprintf('The number of non-black pixels in this image is :');
disp(nbp);
fprintf('Second, calculation with correspoding ground truth image :\n');
img2= imread('D:\Privat\pvjfiles\thesiswork\data\cracks\testingcases\easycase\DSC01633_manualCrack_radial_gt_1369x.jpg');
img2 = rgb2gray(img2);
[y2,x2] = size(img2);
fprintf('\nVerifying if dimensions of both images are same.\n');
if (isequal(size(img),size(img2))==1)
    I_double2 = im2double(img2);
    blackpixels_index2 = find(I_double2==0);
    fprintf('The number of black pixels in this image is :');
    bp2= size(blackpixels_index2);
    disp(bp2(1,1));
    nbp2=y*x-bp2(1,1);
    fprintf('The number of non-black pixels in this image is :');
    disp(nbp2);
    error_percentage=((nbp-nbp2)/(y*x))*100; % we know there are many false positive cases, from working/results experience
    fprintf('The error percentage in this image is : ');
    disp(error_percentage);
    fprintf('The accuracy obtained on this crack-centerline output image is : ');
    disp(100-error_percentage);    
    fprintf('end------------------------------------------------------------');
else
   disp('img and img2 are not the same size.')
   fprintf('\nstop------------------------------------------------------------');

end
% for us error definition is the proportion of extra non black pixels in crack result image as compared to the correspoding
% ground truth, to that of all pixels in the image. in other words, % of
% false positive cases as compared to original ground truth image.


