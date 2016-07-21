function pixelwise_testing
fprintf('\nstart----------------------------------------------------------\n');
fprintf('First , input crack-centerline output image  \n');
img= imread('D:\Privat\pvjfiles\thesiswork\results\cracks\Results of easy case\Updated results\copy for testing\E14_crack.jpg');
img = rgb2gray(img);
same=0;
false_positive_counter=0;
false_negative_counter=0;
[y,x] = size(img);
I_double = im2double(img);

fprintf('Second, input correspoding ground truth image \n');
img2= imread('D:\Privat\pvjfiles\thesiswork\results\cracks\Results of easy case\Updated results\copy for testing\E14_manualCrack_radial_gt.jpg');
img2 = rgb2gray(img2);
[y2,x2] = size(img2);
fprintf('\nVerifying if dimensions of both images are same.\n');
if (isequal(size(img),size(img2))==1)
    I_double2 = im2double(img2);
 
    for i=1:size(I_double,1)
    for j=1:size(I_double,2)
        % get pixel value
        pixel=I_double(i,j);% output img
        pixel2=I_double2(i,j);% gt image
          % check pixel value and assign new value
          if (pixel==pixel2)
              same=same+1;
          else
              %check for false positive and false negative!
              if(pixel > pixel2)
                  false_positive_counter=false_positive_counter+1;
              else
                  false_negative_counter=false_negative_counter+1;
              end
          end       
    end
    end
    different_pixels=(x*y)-same;
    pixelwise_error_percentage=(different_pixels/(x*y))*100;
    pixel_hit_percentage= 100-pixelwise_error_percentage;
    fprintf('pixel_hit_percentage is : ');
    disp(pixel_hit_percentage);
    
    false_positive_percentage=(false_positive_counter/(x*y))*100;
    fprintf('\n False_positive_percentage is :  ');
    disp(false_positive_percentage);
    
    false_negative_percentage=(false_negative_counter/(x*y))*100;
    fprintf('\n False_negative_percentage is :  ');
    disp(false_negative_percentage);
    fprintf('end------------------------------------------------------------');
else
   disp('img and img2 are not the same size.')
   fprintf('\nstop------------------------------------------------------------');
end