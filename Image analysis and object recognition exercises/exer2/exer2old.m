function exer2
I=imread('input.png');
%figure ,imshow(I)
Igr=rgb2gray(I);
e=2.7182; pi=3.142;sigma=0.9;
 % size(Igr) //355,255
 %% size_filter=2 %% |3*sigma|
cx=[-2,-1,0,1,2;-2,-1,0,1,2;-2,-1,0,1,2;-2,-1,0,1,2;-2,-1,0,1,2];
cy=(cx)';
 
 [Gog_x_mask,Gog_y_mask]=gog_filter_mask(cx,cy,sigma,pi,e);
 [Ix,Iy]=convolution(Igr,Gog_y_mask,Gog_x_mask);
 [M,W,Q]=auto_corr(Ix,Iy);
 [Mc,Wbar,Qbar,peaks]=potentialMask(W,Q); 
 
 
 %%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%
 
 function[Gog_x_mask,Gog_y_mask]=gog_filter_mask(cx,cy,sigma,pi,e)
        for i=1:5
             for j=1:5

                 Gog_x_mask(i,j)=double((-1*(cx(i,j)/(2*pi*sigma^4))).*e^(-1*((cx(i,j).*cx(i,j)+cy(i,j).*cy(i,j))/(2*sigma^2))));
             end
        end
            figure ,imshow(Gog_x_mask,[]) %1

            Gog_y_mask=(Gog_x_mask)';

           figure ,imshow(Gog_y_mask,[]) %2

 end      
 %%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%   
 
 function[Ix,Iy]=convolution(Igr,Gog_y_mask,Gog_x_mask)
    Ix=zeros(355,255);
    Iy=zeros(355,255);
    Ix=double(Ix);
    Iy=double(Iy);
    % A=zeros(5,5);
    for m=3:353
      for n=3:253
          A = Igr(m-2:m+2,n-2:n+2);
          
          Ix(m,n)=sum(sum(double(Gog_x_mask.*double(A))));
           Iy(m,n)=sum(sum(double(Gog_y_mask.*double(A))));
      end
    end
    figure ,imshow(Ix,[]) %3
    figure ,imshow(Iy,[]) %4
  Gradient = sqrt((Ix).^2 + (Iy).^2);
  figure ,imshow(Gradient) %5
 end
end

 %%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%
 
 function [M,W,Q]=auto_corr(Ix,Iy)

  LX = Ix.*Ix;
  LY = Iy.*Iy;
  LXY = Ix.*Iy;
     for m=3:353
      for n=3:253
        B = LX(m-2:m+2,n-2:n+2);   
        M_LX=sum(sum(double(B)));
        C = LY(m-2:m+2,n-2:n+2);   
        M_LY=sum(sum(double(C)));
        D = LXY(m-2:m+2,n-2:n+2);   
        M_LXY=sum(sum(double(D)));
        M=[M_LX,M_LXY;M_LXY,M_LY;];
        
         W(m,n) = (trace(M)/2) - sqrt((trace(M)/2)^2-det(M));
        Q(m,n) = 4*det(M)/(trace(M)^2);
      end
     end
% size(W) how can the size of W and Q be 353,253? I think it should be 351,251!!! LOOK AT THE LOOP COUNTER!
%  size(Q)
figure;imshow(W);colormap(jet); %%% fig 6
figure;imshow(Q);colormap(jet); %%% fig 7
  end
  
 %%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%
 
 function [Mc,Wbar,Qbar,peaks]=potentialMask(W,Q)
 tw=0.004;
 tq=0.5;
 Mc=zeros(353,253);
 Wbar=zeros(353,253);
 for i=1:353
     for j=1:253
       if(W(i,j)>tw && Q(i,j)>tq)
           Mc(i,j)=1;
       else
           Mc(i,j)=0;
       end
    end
 end
 figure ,imshow(Mc,[]) %8
 Qbar=Q.*Mc
 Wbar=Qbar.*Mc;
  peaks = houghpeaks(double(gray2ind(Wbar)), 40, 'threshold', double(ceil(0.3*max(Wbar(:)))) );
   figure ,plot(peaks) %9
 end