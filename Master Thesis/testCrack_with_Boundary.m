function testCrack_with_Boundary

for a = 2:5
   filename = ['A' num2str(a,'%2d') '.jpg'];
   imgC = imread(filename);
   imgC_double= im2double(imgC);
   luv_imag2= rgbConvert(imgC_double,'luv');

   bound_img = predictBoundaries(luv_imag2,weakLearners,p)
   figure(a);
   imshow(bound_img,[])
 
   [out_score_max,out_scale_max ] = max(bound_img,[],p.dimension+1); % get maximum along scales 
   nms_score =edgeNms(out_score_max,[],1,5); % apply non-maxima suppression to extract centerlines only
   figure(a*5);
   imshow(nms_score,[]);
   
end

end