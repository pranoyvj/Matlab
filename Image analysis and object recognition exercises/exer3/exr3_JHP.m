%% yoyo~pvj

function exr3_JHP
Img=imread('inputex3.jpg');
Igr=mat2gray(mean(Img,3));
figure (1),imshow(Igr);
%% size(Igr)=(370,500)
%%constants
e=2.7182; ;sigma=0.5;
cx=[-2,-1,0,1,2;-2,-1,0,1,2;-2,-1,0,1,2;-2,-1,0,1,2;-2,-1,0,1,2];
cy=(cx)';
%%

[Gog_x_mask,Gog_y_mask]=gog_filter_mask(cx,cy,sigma,pi,e);
[Ix,Iy,Gradient]=convolution(Igr,Gog_y_mask,Gog_x_mask);
[Bin_Img,edge_pixels] = Binarization(Gradient);
[H_map,theta,rho,peak]=Hough_line_detection(Bin_Img,Ix,Iy,edge_pixels, Igr);

end %% main end

%%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%

    function[Gog_x_mask,Gog_y_mask]=gog_filter_mask(cx,cy,sigma,pi,e)
        for i=1:5
             for j=1:5

                 Gog_x_mask(i,j)=double((-1*(cx(i,j)/(2*pi*sigma^4))).*e^(-1*((cx(i,j).*cx(i,j)+cy(i,j).*cy(i,j))/(2*sigma^2))));
             end
        end
            figure (2),imshow(Gog_x_mask,[]) %1

            Gog_y_mask=(Gog_x_mask)';

           figure (3),imshow(Gog_y_mask,[]) %2
        
    end   

%%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%

    function[Ix,Iy,Gradient]=convolution(Igr,Gog_y_mask,Gog_x_mask)
    Ix=zeros(370,500);
    Iy=zeros(370,500);
    Ix=double(Ix);
    Iy=double(Iy);
    % A=zeros(5,5);
    for m=3:368
      for n=3:498
          A = Igr(m-2:m+2,n-2:n+2);
          
          Ix(m,n)=sum(sum(double(Gog_x_mask.*double(A))));
           Iy(m,n)=sum(sum(double(Gog_y_mask.*double(A))));
      end
    end
    figure (4),imshow(Ix,[]) 
    figure (5),imshow(Iy,[]) 
    Gradient = sqrt((Ix).^2 + (Iy).^2);
    figure (6),imshow(Gradient) 
    
    end

%%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%

        function [Bin_Img,edge_pixels] = Binarization(Gradient)
                 thresh=0.06; % threshold assumed/approximated using mean value intensity of image 
                 Bin_Img=zeros(370,500);
                 edge_pixels=[];
                 for i=1:370;
                 for j=1:500;                            
                    if (Gradient(i,j)<thresh)               
                        Bin_Img(i,j)=0.0;              
                    else                                 
                        Bin_Img(i,j)=1.0; 
                        new_row =[i, j];
                        edge_pixels=[edge_pixels;new_row]; %% edge pixels is 2 element column vector!!           
                    end
                 end
                 end
                figure(7), imshow(Bin_Img);
               % size(edge_pixels)=    14196           2
               %% last new row "entry" in edge pixels is 352 and 252...so index is less than 370.
        end
%%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%


    function [H_map,theta,rho,peak]=Hough_line_detection(Bin_Img,Ix,Iy,edge_pixels, Igr)
        
        
        t=-90:89;
        rho_max=2*sqrt(370^2+500^2);
        %rho=zeros(1,rho_max);
          H_map=zeros(round(rho_max),180);
        r=[-floor(rho_max-rho_max/2):floor(rho_max-rho_max/2)];  %% used bcoz if we use 0...max   then error while using houghlines(so here -623:+622)
        theta_grad=atan(Iy./Ix); 
        %% size(theta_grad) is 370,370    IS this  correct?
        
        for m=1:14196 ;%% I use 14196 bcoz that many rows are there in "edgle_pixels". Each row has i,j index/pixel.
            
                    
   %%ERROR IN THIS LINE%%         
   rho=edge_pixels(m,2)*cos(theta_grad(edge_pixels(m,1),edge_pixels(m,2)))+edge_pixels(m,1)*sin(theta_grad(edge_pixels(m,1),edge_pixels(m,2)));
                   %% eqn as in slides 

                    theta=theta_grad(edge_pixels(m,1),edge_pixels(m,2));%% returns the answer in readians ...so need to convert to degree
                    theta=round(theta*(180/pi));
                   
                    ind_r=find(r==round(rho));
                    
                    ind_t=find(t==round(theta));
                    
                    
                    H_map(ind_r,ind_t)=H_map(ind_r,ind_t)+1;
        end
        
        figure (8),imshow(H_map);
        
          peak = houghpeaks(double(H_map), 40, 'threshold',double(ceil(0.03*max(H_map(:)))) );
          hold on;
         
          plot(peak(:,2), peak(:,1), 's', 'color', 'red');
          %hold off;
        
        figure (9),imshow(Igr);  
        lines=houghlines(Bin_Img,t,r,peak);
        hold on;
        for k = 1 : length(lines)
            xy = [lines(k).point1; lines(k).point2];
            plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
            plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
            plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'red');
        end
    hold off;
        
    end

    
%%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%
















