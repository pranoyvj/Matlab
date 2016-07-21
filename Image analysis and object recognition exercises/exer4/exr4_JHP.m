%% yoyo~pvj

function exr4_JHP
Img=imread('taskA.png');
Igr=mat2gray(mean(Img,3));
figure (1),imshow(Igr);
%%size(Igr) is  678        1024

J_noisy=imnoise(Igr,'gaussian',0,0.01);
figure (2),imshow(J_noisy);

[guas_filter]= two_dgaussian(pi); %%func call
[img_fft,filter_fft]=freq_dom(guas_filter,J_noisy,Igr);%%func call
[Df,Df2]=bin_mask();

end
 %%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%
 
 %%function to create 2d gaussian filter
 function[guas_filter]= two_dgaussian(pi)
     e=2.7182; sigma=5.0;
     
     r = round(3*sigma); cx = -r:r;
     cx = repmat(cx, numel(cx),1); 
     
     cy=(cx)';
     for i=1:r*2+1
             for j=1:r*2+1

                 guas_filter(i,j)=double((1/(2*pi*sigma^2)).*e^(-1*((cx(i,j).*cx(i,j)+cy(i,j).*cy(i,j))/(2*sigma^2))));
             end
     end
            figure (3),imshow(guas_filter, []); 
 end

 %%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%
  %% now covolution in spatial domain that is = multiplication in freq domain
 function[img_fft,filter_fft]=freq_dom(guas_filter,J_noisy,Igr)
 
 s_img = size(J_noisy);
 s_filter = size(guas_filter);
 filter_pad= padarray(guas_filter, s_img-s_filter, 0, 'post');
 
 B_shift=circshift(filter_pad,-floor(0.5*size(guas_filter)));
 % figure ,imshow(B_shift);
 img_fft=fft2(J_noisy); %converting noisy image to Dft in freq domain 
 figure(4);
 imagesc(log(abs(img_fft)));% code to plot to get o/p as in slide
 
 filter_fft=fft2(B_shift);%converting circ-shifted filter to Dft in freq domain 
 figure (5),imshow(filter_fft);
 colormap(jet);
 
 multi_freq_domain=img_fft.*filter_fft;
 smoothed_img=ifft2(multi_freq_domain);
 figure (6),imshow(smoothed_img);
 
 figure(7);
 imagesc(log(abs(fftshift(fft2(Igr)))));%original image
 figure(8);
 imagesc(log(abs(fftshift(img_fft))));% noisy image
 figure(9);
 imagesc(log(abs(fftshift(fft2(guas_filter)))));% guas filter
 figure(10);
 imagesc(log(abs(fftshift(fft2(smoothed_img)))));
  
 end

%%%--------------------------------------------------------------------------------------------------------------------------------------------------------%%%

function[Df,Df2]=bin_mask()

%%%---------------------------------- for image training B
tran1=imread('trainingB.png');
tran1gr=mat2gray(mean(tran1,3));
tlevel=graythresh(tran1gr);
BW=logical(im2bw(tran1gr,tlevel));

[B,L,N] = bwboundaries(BW);
figure (11); imshow(BW); hold on;
for k=1:length(B),
    boundary = B{k};
    if(k > N)
        plot(boundary(:,2),...
            boundary(:,1),'g','LineWidth',2);
    else
        plot(boundary(:,2),...
            boundary(:,1),'r','LineWidth',2);
    end
end

D=boundary(:,1)+i*boundary(:,2);
D_transpose=(D)';
Df=fft(D_transpose);
n=24;
Df=Df(2:(n+1));% translation invariance 
Df=Df/(Df(1));% scale invariance
Df=abs(Df)%orientation invariance

%%%---------------------------------- for image test1B.jpg

tran2=imread('test1B.jpg');
tran2gr=mat2gray(mean(tran2,3));
%figure ,imshow(tran2gr);
t2level=graythresh(tran2gr);
BW2=logical(im2bw(tran2gr,t2level));

[B2img,L2,N2] = bwboundaries(BW2, 'noholes');
figure (12); imshow(BW2);

figure (13),imshow(tran2gr);hold on;
for k2=1:length(B2img),%upto 51
   
    boundary2 = B2img{k2};
    %size(boundary2)= 685 2
   D2img=boundary2(:,1)+i*boundary2(:,2);%complex numbers
    
        rs=size(D2img)
        if rs(1) > n        
            D_transpose2=(D2img)';
            Df2=fft(D_transpose2);
            size(Df2)
            n=24;
            Df2=Df2(2:(n+1));% translation invariance 
            Df2=Df2/abs(Df2(1));% scale invariance
            Df2=abs(Df2);%orientation invariance
            if norm(Df-Df2)<0.06
            plot(boundary2(:,2), boundary2(:,1));
            end       
        end
end

%%%---------------------------------- for image test2B.jpg

tran3=imread('test2B.jpg');
tran3gr=mat2gray(mean(tran3,3));
t3level=graythresh(tran3gr);
BW3=im2bw(tran3gr,t3level);
figure (14),imshow(BW3);

[B3img,L3,N3] = bwboundaries(BW3, 'noholes');
figure (15); imshow(tran3gr); hold on;
for k3=1:length(B3img),%upto 51
   
    boundary3 = B3img{k3};
 
        D3img=boundary3(:,1)+i*boundary3(:,2);%complex numbers
    
        rs2=size(D3img)
        if rs2(1) > n
        D_transpose3=(D3img)';
        Df3=fft(D_transpose3);
        n=24;
        Df3=Df3(2:(n+1));% translation invariance 
        Df3=Df3/abs(Df3(1));% scale invariance
        Df3=abs(Df3);%orientation invariance
            if norm(Df-Df3)<0.06
            plot(boundary3(:,2), boundary3(:,1));
            end       
        end
    end
end









