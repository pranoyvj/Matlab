function Fujita_preprocessing
I = rgb2gray(imread('D:\Privat\pvjfiles\thesiswork\data\cracks\testingcases\easycase\DSC01633.JPG'));
figure(1); imshow(I);                     title('Originalbild'); 
figure(2); imshow(test(I, 20, 1, 0.1));  title('Fujita (m = 20, s = 1, t = 0.1)'); 

function J = test(I, m, s, t) % Fujita 2006 & Eugen Müller 
%            =================
subtr = imsubtract(medfilt2(I, [m m]), I); % extract high image frequencies
gaussian = imfilter(subtr, fspecial('gaussian', wins(s), s));   % smoothing
J = im2bw(gaussian, t);                                      % thresholding

function w = wins(s) 
w = 2*ceil(3*s)+1;                    % window size from standard deviation