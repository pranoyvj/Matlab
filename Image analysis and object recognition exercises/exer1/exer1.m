%% yoyo~pvj

function exer1
    I = imread('input.jpg');%---- read input image
    
    Igr=histogram(I);
    New_Igr=streching(Igr);  
    Bin_Img =Binarization(New_Igr);
    [dil,Close_IM]=morphological_refinement(Bin_Img);
    fin=final_image(Close_IM,New_Igr);
end
   
    function Igr=histogram(I)
   
    %img_mean = mean(mean(I));%---- compute mean of pixels in image
    %img_mean;
    Igr=rgb2gray(I);%---- function to convert the given RGB image to Grayscale Image
    % imshow(Igr);
    % size(I) // 691*1048
    figure(1), imhist(Igr,256);%1---- function to o/p histogram og resulting grayscale image
    
    %---- Characteristics of Histogram:
    %    1> In the resulting grayscale image there are no pixels ,with
    %       corresponding intensity values from 0-150.
    %    2> Large number of pixels lie in the intensity range of 150-170
    %    3> There are no pixels,having corresponding intensity values from 220-255.
    %----
    end
    
    
  %%%-------------------------------------------------------------------------------------------%%%%                  
 
                    
     function New_Igr=streching(Igr) %% contrat streching/ thresholding
            dIgr=double(Igr);
            mini=min(min(dIgr));%% two time min/max bcoz min/max fuction outputs min value of each column in 2d array matrix
            maxi=max(max(dIgr));
            New_Igr = zeros(691,1048); %--- self code for streching %%%%%

                     New_Igr=((dIgr-mini)/(maxi-mini));


            figure(2), imshow(New_Igr); %2
            figure(3), imhist(New_Igr,256); %3
            %--- the resulting image New_Igr has more contrast in comparison to
            %    previous image.The Histogram gets streched.
            size(New_Igr)
    end 
 
 
  %%%-------------------------------------------------------------------------------------------%%%%         
  
  
  function Bin_Img = Binarization(New_Igr)
                 thresh=0.3; % threshold assumed/approximated using mean value intensity of image 
                 Bin_Img=zeros(691,1048);
                 for i=1:691;
                 for j=1:1048                            % It was time consuming to find the threshold. In the lecture 
                    if (New_Igr(i,j)<thresh)             % slides there is no criteria for selecting the threshold.  
                        Bin_Img(i,j)=255.0;              % Searched online but there was no clear or no 
                    else                                 % easy to understand criteria for selecting threshold. 
                        Bin_Img(i,j)=0.0;                % hAD TO find threshold by TRIAL and Error.
                                    
                    end
                 end
                 end
                figure(4), imshow(Bin_Img); %4
  end
        
  
     %%%-------------------------------------------------------------------------------------------%%%%   
     
    
    function [dil,Open_IM,Close_IM]=morphological_refinement(Bin_Img)         
        %%%%% Morphological operators OPENING AND
        %%%%% CLOSING%%%%%%%%%%%%%%%%%%%%%%
            
         SE=strel('disk',4); %% create structuring element
         Open_IM = imopen(Bin_Img,SE) %%% opening 
         %figure(5), imshow(Open_IM);
         Close_IM = imclose(Open_IM,SE) %%%closing 
         figure(5), imshow(Close_IM);
         
        %%% Dilation by direct MATLAB function
        % Dilation = imdilate(Bin_Img,SE);
        %  figure, imshow(Dilation);                      
        %%there are diffrences in the o/p of matlab in buillt and self
        %code implementaion of dilation
        %% Opening is erosion + dilation
        %% closing is dilation + erosion
                                                        
         %% ---- DILATION BY SELF CODE
         mask=logical(ones(3,3));
         for m=2:690
         for n=2:1047
             A=[Bin_Img(m-1,n-1)    ,   Bin_Img(m-1,n)  ,   Bin_Img(m-1,n+1);
                Bin_Img(m,n-1)      ,   Bin_Img(m,n)    ,   Bin_Img(m,n+1);
                Bin_Img(m+1,n-1),       Bin_Img(m+1,n)    , Bin_Img(m+1,n+1) ];
                Z=intersect(mask(:),logical(A(:)/255)); 
        if any(Z)==0
        dil(m,n)=0;
        else
            dil(m,n)=1;
        end
         end
        end
        figure(6), imshow(logical(dil)); 
    end    
        
    %%%-------------------------------------------------------------------------------------------%%%%        
    
    
        function fin=final_image(Close_IM,New_Igr)  
    %%%%--final-image after combine the intensity stretched image with opening and closing result
            p=0;q=0;   
            for p=1:691
              for q=1:1048
               if Close_IM(p,q)==255
                  fin(p,q)=255.0;
               else
                  fin(p,q)=New_Igr(p,q);
               end
              end
            end
            figure(7), imshow(fin);       
        end
     
     %%% for study and learning purpose results are satisfactory. But for
      %%% advanced research purposes, where accuracy is important, these methods may be
      %%% unimpressive, or may fail to acheive good results. These methods may
      %%% fail under certain situations.