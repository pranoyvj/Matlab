% ~yoyo pvj
% group JHP

% sample solution for exercise 5 in the course 
% "Image Analysis and Object Recognition"

% Input: 2 imagefiles with one or more channels covering the same field of
% view from different perspectives

% The code coducts the following tasks: 
% Implementations are all based on the paper (Lowe, 2004): "Distinctive
% Image Features from Scale-Invariant Keypoints"
% Steps of the workflow:
% 1) Calculation of a DoG-Pyramid using differences of Gaussians
% 2) Localization of maxima and minima in the DoG-Pyramid
% 3) Refinement of keypoint positions
% 4) Filtering of keypoints using contrast and edge-iness information
% 5) Precompute orientations per pixel in all images (angles and magnitudes)
% 6) Determine main orientations for all keypoints
% 7) Computation of the descriptor (128-element vector for each point)
%
% 8) Do steps 1-7 for the two input images
% 9) Match the points using euclidian distance measure
% 10) Visualize the results


% Main function: TestSIFTFeatures
% Here the whole processing chain is done
% Used functions: 
%   - CalcSIFTFeatures to do steps 1-7 for two input images
%   - MatchSIFTFeatures to do step 9
%   - VisualizeMatches to do step 10
function TestSIFTFeatures

    % select and import image data
    [file1, path1, image1, img_size1] = read_image('Select file 1 for processing');
    image1 = mat2gray( image1 );
    [file2, path2, image2, img_size2] = read_image('Select file 2 for processing');
    image2 = mat2gray( image2 );
    
    % calculate SIFT features
    plot = 0;   % 0 --> no plotting of intermediate results; 1 --> plotting activated
    pre_smooth = 1; % 0 --> no prior smoothing; 1 --> prior smoothing (more robustness)
    thresh_contrast = 0.03;%; % the higher, the less points - default: 0.03
    thresh_edges = 6; %the lower, the less points - default: 10
    
    [Keypoints1, Locations1, image1] = CalcSIFTFeatures(image1, img_size1, plot, pre_smooth, thresh_contrast, thresh_edges);
    [Keypoints2, Locations2, image2] = CalcSIFTFeatures(image2, img_size2, plot, pre_smooth, thresh_contrast, thresh_edges);

    if ~length(Keypoints1(:,1)) | ~length(Keypoints2(:,1))
        return;
    end
    % match points
    thresh_dist = 0.2; % all matches [0,...,sqrt(128)] gt thres_dist will be dismissed 
    [matches] = MatchSIFTFeatures(Keypoints1, Keypoints2, thresh_dist);
    
    % visualize result
    VisualizeMatches(image1, image2, matches, Locations1, Locations2)
end

%-------------------------------------------------------------------------
% Computation of the final descriptor --> Task for assignment 5
% Inputs:
%   - Magnitude_Pyramid: Cell with length(Magnitude_Pyramid) = length(Maxima)
%       --> in each level for each pixel the local gradient magnitudes are stored
%   - Angle_Pyramid: Cell with length(Angle_Pyramid) = length(Maxima)
%       --> in each level for each pixel the local gradient orientations are stored
%   - Maxima: Cell with maxima in each scale step-->array of size (n_points, 5)
%             [y,x,sigma,orientation,magnitude]
%   - Minima: see maxima
%   - Sigmas: vector of size (n_scales) with corresponding values sigma for
%             each scale level of DoG-pyramid
%   - img_size: size of the image [y,x]
% Outputs:
%   - Keypoints: Array with keypoint information for the image
%                 Dimensions: [n_keypoints, 128]
%   - Locations: Coordinates of keypoints for the image; in
%                 each row of that array (size [n_points, 5]) we have:
%                 [y,x,sigma,orientation,magnitude]
% The Keypoits- and the Locations-array will have the same number of rows
% and a row in Keypoits(i) corresponds to a row Locations(i)!
% After computation there will be not longer a separation of minima and
% maxima!
function [Keypoints, Locations] = ComputeDescriptor(Magnitude_Pyramid, Angle_Pyramid, Extrema, Sigmas, img_size)
    
     % to create 2d gaussian filter
     e=2.7182; sigma1=8.0;
     
     r = round(3*sigma1); cx = -r:r;
     cx = repmat(cx, numel(cx),1); 
    
     Keypoints=[];
     Locations=[];
     
     cy=(cx)';
     for i=1:r*2+1
             for j=1:r*2+1

                 guas_filter(i,j)=double((1/(2*pi*sigma1^2)).*e^(-1*((cx(i,j).*cx(i,j)+cy(i,j).*cy(i,j))/(2*sigma1^2))));
             end
     end
            figure (1),imshow(guas_filter, []); 
       
            for i=1:length(Sigmas)
                
                extrema = Extrema{i};
                for j=1:size(extrema, 1)
                
                ex = extrema(j,:);    
                    
                counter=1
                xc=repmat([-7.5:7.5],[16,1]);
                yc=xc';
                
                thetam = ex(4);
                %thetam=j(counter,4)
                theta1=(thetam-90);
                %rotation
                n_xc=sind(theta1)*yc+cosd(theta1)*xc;
                n_yc=cosd(theta1)*yc-sind(theta1)*xc;
                %% Determine nearest scale Sigma n
                %% Nearest scale sigma n: Already available . Where is it?
                scale_index = i;
                
                m_nxc=round(n_xc+49/2);
                m_nyc=round(n_yc+49/2);
                counter=counter+1;
                
                gauss_indices = sub2ind(size(guas_filter), m_nyc, m_nxc);
                guas_filter_resampled = guas_filter(gauss_indices);
                
                
                %% Do resampling for magnitude and angles
                m_nxc=round(n_xc+ex(2));
                m_nyc=round(n_yc+ex(1));
                
                % check, if translated coordinates fall into image
                
                if( min(m_nxc(:)) <1 | min(m_nyc(:))<1 | max(m_nxc(:))>img_size(2)| max(m_nyc(:))>img_size(1) )
                    continue;
                end
                
                
                image_indices = sub2ind(img_size, m_nyc, m_nxc);
                
                % sample gradient angles and magnitudes
                mag = Magnitude_Pyramid{scale_index}(image_indices);
                angle = Angle_Pyramid{scale_index}(image_indices);
                
                % weighted gradient magnitude
                mw=guas_filter_resampled.*mag;
                
                % Rotate gradient directions
                angle_sampled_rotated=angle-thetam;

                % check, if angles are 0<angle<360
                
                ind2=find(angle_sampled_rotated>360);
                angle_sampled_rotated(ind2)=angle_sampled_rotated(ind2)-360;
                
                ind2=find(angle_sampled_rotated<0);
                angle_sampled_rotated(ind2)=angle_sampled_rotated(ind2)+360;
                
                
                
                
                
                % build histograms
                histograms = zeros(16,8);
                h_counter = 1;

                % 8 bins -> 45 degrees per bin
                binsize = 45;
                bins = [0:binsize:350];
                for hy = 1:4
                    % local y coordinates of actual window
                    y_start = 1 + 4 * (hy - 1);
                    y_stop = y_start + 3;
                    for hx = 1:4
                        % local x coordinates of actual window
                        x_start = 1 + 4 * (hx - 1);
                        x_stop = x_start + 3;
                        
                        % extract the corresponding subcells
                        angle_cell = angle_sampled_rotated(y_start:y_stop, x_start:x_stop);
                        magni_cell = mw(y_start:y_stop, x_start:x_stop);
                        angle_cell_r = round(angle_cell);
                        
                        % build histogram with loop over all pixels in cell
                        for j = 1:length(angle_cell)^2
                            % determine, in which bin actual element falls
                            [h,centers] = hist(angle_cell_r(j), bins+binsize/2);
                            pos = find(h);
                            % and add corresponding magnitude
                            histograms(h_counter, pos) = histograms(h_counter, pos) + magni_cell(j);
                        end
                        h_counter = h_counter + 1;
                    end
                end
                
                d=reshape(histograms, 1, 128);
                d=d./(norm(d));
                ind=find(d>0.2)
                d(ind)=0.2;
                
                d=d./(norm(d));
                
                Keypoints = [Keypoints;d];
                Locations = [Locations;ex];
                
                end
                
            end


end


%--------------------------------------------------------------------------

% Calculation of SIFT features for an input image
% Steps 1-7 are done by this function (and subfunctions)
% Input parameters:
%    - image: image array (one channel)
%    - img_size: size of image array [y,x]
%    - plot: boolean number: 0-->no plot of results; 1-->plot results
%    - pre_smooth: boolean number: 0-->no pre-smoothing of input image; 
%                                  1-->pre-smoothing is done (image is doubled in size before!!!)
%    - thresh_contrast: keypoint candidates with a contrast measurement
%      lower this value are rejected (default value: 0.03 (Lowe, 2004))
%    - thresh_edges: keypoint candidates with an edge-iness measurement
%      higher this value are rejected (default value: 10 (Lowe, 2004))
% Output parameters:
%    - Keypoints: Arrays with keypoint information for the image
%                 Dimensions: [n_keypoints, 128]
%   - Locations: Coordinates of keypoints for the image; in
%                 each row of that array (size [n_points, 5]) we have:
%                 [y,x,sigma,orientation,magnitude]
%   - Image: when pre_smooth=1 was set, the image is resized to be 4xbigger
%            than before (reason: smoothing with sigma = 1.6 leads to loss of details otherwise)
%            --> Image is output in order to provide it for plotting
%   - Used Functions (steps 1-7):
%       - CalcDoGPyramid
%       - FindLocalExtrema
%       - PlotExtrema
%       - LocalizeContrastEdges
%       - ComputeOrientations
%       - AssignOrientations
%       - ComputeDescriptor
function [Keypoints, Locations, Image] = CalcSIFTFeatures(image, img_size, plot, pre_smooth, thresh_contrast, thresh_edges)
    
    %----------------------------------------------------------------------
    % Calculate scale space and differences of Gaussians
    % standard deviations
    
    sigma_start = 2.0;
    % cover the whole scale space -> depends on image size
    sigma_stop = 2^round(log2(min(img_size(1:2))));

    % three steps per octave (1 octave = doubling sigma) are optimal
    % so calculate here the overall number of intervals in scale space
    % based on sigma_start and sigma_stop
    n_int = ceil( (log(sigma_stop)/log(2)-log(sigma_start)/log(2)) * 3 );
    
    %----------------------------------------------------------------------
    % Calculate scale space and differences of Gaussians
    % caution: pree-smoothing leads to an doubled image size (nn-sampling)
    [DoG_pyramid, Sigmas, Image_pyramid, img_size] = CalcDoGPyramid(n_int, image, sigma_start, sigma_stop, plot, pre_smooth);
    % when pre-smoothing was done: image has double size now!
    % The image is stored in the first cell of the Image pyramid the other 
    % images are blurred versions
    Image = Image_pyramid{1};
    
    %----------------------------------------------------------------------
    % find initial local maxima and minima in DoG Images    
    [Maxima, Minima] = FindLocalExtrema(DoG_pyramid, Sigmas);
    
    if plot
        % plot results (use Image_pyramid{1} instead of image due to possibly 
        % changed image dimensions caused by pre-smoothing)    
        title = 'Maxima (red) and Minima (green) in GoG scale space';
        PlotExtrema(Maxima, Minima, Image_pyramid{1}, title);
    end
    
    %----------------------------------------------------------------------
    % refine the results
    
    % at first just do shifting of extrema, if position tells us to do so
    shifting = 1; % therefore, just set shifting to 1
    % this avoids the refinement and just shifts yx positions if necessary
    Maxima = LocalizeContrastEdges(Maxima, DoG_pyramid, Sigmas, thresh_contrast, thresh_edges, shifting);
    Minima = LocalizeContrastEdges(Minima, DoG_pyramid, Sigmas, thresh_contrast, thresh_edges, shifting);

    % now, we can do refinement with new positions (contrast and edge-iness)
    shifting = 0; % no shifting, just filtering out due to contrastt and edge-iness
    Maxima = LocalizeContrastEdges(Maxima, DoG_pyramid, Sigmas, thresh_contrast, thresh_edges, shifting);
    Minima = LocalizeContrastEdges(Minima, DoG_pyramid, Sigmas, thresh_contrast, thresh_edges, shifting);
    
    % plot results    
    if plot
        title = 'Maxima (red) and Minima (green) after refinement';
        PlotExtrema(Maxima, Minima, Image_pyramid{1}, title);
    end
    
     %----------------------------------------------------------------------
     % orientation assignment: compute orientations for each scale, where
     % extrema occur
     [Magnitude_Pyramid, Angle_Pyramid] = ComputeOrientations(Image_pyramid, Maxima, Minima);
     
    %----------------------------------------------------------------------
    % assign orientations
    [Maxima] = AssignOrientations(Magnitude_Pyramid, Angle_Pyramid, Maxima, Sigmas);
    [Minima] = AssignOrientations(Magnitude_Pyramid, Angle_Pyramid, Minima, Sigmas);

    % plot results    
    if plot
        title = 'Refined maxima (red) and Minima (green) with orientation and magnitude.';
        PlotExtrema(Maxima, Minima, Image_pyramid{1}, title);
    end
    %----------------------------------------------------------------------
  
    % We are not longer interested whether the points are max or mins
    % --> concatenation to one array each scale
    for s = 1:length(Maxima)
       Extrema{s} = [Maxima{s}; Minima{s}]; 
    end
    
    % descriptor computation
    [Keypoints, Locations] = ComputeDescriptor(Magnitude_Pyramid, Angle_Pyramid, Extrema, Sigmas, img_size);
    
end


%--------------------------------------------------------------------------

% Function to visualize the matching-results using the two input images
% Input parameters:
%   - image1, image2: arrays (rows,columns) with input images
%   - matches: array with matching results, in each row of that array we have 3 quantities: 
%      [ index_of_point_in_image1, index_of_point_in_image2, distance_measure ]
%   - Locations1, Locations2: Coordinates of keypoints for image1 and 2; in
%     each row of that array (size [n_points, 5]) we have:
%      [y,x,sigma,orientation,magnitude]
function VisualizeMatches(image1, image2, matches, Locations1, Locations2)
    
    % get screensize
    scrsz = get(0, 'ScreenSize');
    img_size1 = size(image1);
    img_size2 = size(image2);
    % compute size of new image which shows both images at once
    s_new = [ max([img_size1(1), img_size2(1)]), img_size1(2)+img_size2(2) ];
    % offset in x for image 2 coordinates
    xoffset_img2 = img_size1(2);
    
    % concatenate images
    image_both = zeros( s_new );
    image_both(1:img_size1(1), 1:img_size1(2)) = image1;
    image_both(1:img_size2(1), xoffset_img2+1:s_new(2)) = image2;
    
    % plot the image(s)
    figure('Position', [0,0,scrsz(3),scrsz(4)],'name', 'Matching results.');
    imshow(image_both);
    
    % plot the lines between matched points
    Y1 = Locations1(matches(:,1),1);
    Y2 = Locations2(matches(:,2),1);
    X1 = Locations1(matches(:,1),2);
    X2 = Locations2(matches(:,2),2);
    X2 = X2 + xoffset_img2;
    hold on
    for l = 1:length(Y1(:,1))
        plot([X1(l), X2(l)], [Y1(l), Y2(l)], 'r-');
    end
    hold off;
end

%--------------------------------------------------------------------------

% Matching of keypoints from 2 different images
% Input parameters:
%    - Keypoints1, Keypoints2: Arrays with keypoint information for image 1 and 2
%      Dimensions: [n_keypoints, 128]
%    - thresh_dist: threshold for euclidean distance measurement
% Output parameters:
%   - matches: array with matching results, in each row of that array we have 3 quantities: 
%      [ index_of_point_in_image1, index_of_point_in_image2, distance_measure ]
function [matches] = MatchSIFTFeatures(Keypoints1, Keypoints2, thresh_dist)

    match_counter = 1;
    % check each combination
    for i = 1:length(Keypoints1(:,1))
        % keypoint image1
        k1 = Keypoints1(i,:);
        min_dist = sqrt(128); % this is the maximum possible value
        
        for j = 1:length(Keypoints2(:,1))
            % keypoint image2
            k2 = Keypoints2(j,:);
            % distance measurement
            d = sqrt( sum( (k1 - k2).^2) );
            % is it the minimum so far?
            if d < min_dist
                % store, if yes
                min_dist = d;
                matches(match_counter, :) = [i,j, min_dist];
            end
        end
        match_counter = match_counter + 1;
    end

    % are there points matched more than once?
    keep = zeros(length( matches(:,1) ),1);
    for j = 1:length( Keypoints2 )
            
        test = find(matches(:,2) == j);
        % if yes:
        if length(test) > 1
            % find the minimum position
            [min_d, min_pos] = min( matches(test,3) );
            % and store it (all others will be removed)
            keep(test(min_pos)) = 1;
        elseif length(test) == 1
            keep(test) = 1;
        end
    end
    % erase doubled matches
    erase = find(~keep);
    if erase
        matches = matches(setdiff(1:size(matches,1),[erase]),:);
    end    
    
    % identify matches to reject due to threshold
    test = find(matches(:,3) > thresh_dist);
    if test
        matches = matches(setdiff(1:size(matches,1),[test]),:);
    end
end



%-------------------------------------------------------------------------
% - Computation of main angles and magnitudes for each keypoint candidate
%   from pre-computed magnitudes and angle pyramids
% - Extension of keypoint information from [y,x,sigma] to [y,x,sigma,orientation,magnitude]
%
% Inputs:
%   - Magnitude_Pyramid: Cell with length(Magnitude_Pyramid) = length(Maxima)
%       --> in each level for each pixel the local gradient magnitudes are stored
%   - Angle_Pyramid: Cell with length(Angle_Pyramid) = length(Maxima)
%       --> in each level for each pixel the local gradient orientations are stored
%   - Extrema: Cell with maxima or minima in each scale step-->array of size (n_points, 3)
%   - Sigmas: vector of size (n_scales) with corresponding values sigma for
%             each scale level of DoG-pyramid
%   - img_size: size of the image [y,x]
% Outputs:
%   - Extrema: Cell with maxima or minima in each scale
%     --> The input cell "Extrema" with arrays of size (n_points, 3) will
%     be extended in that function to arrays of the size (n_points, 5)!!!
function [Extrema] = AssignOrientations(Magnitude_Pyramid, Angle_Pyramid, Extrema, Sigmas)
    
    % radius of sampling window
    r = 4;
    % corresponding window size
    window_size = r*2+1;
    
    % loop over all levels with extrema
    for i = 1:length(Extrema)
        if length(Extrema{i})
            
            % size of input image
            image_size = size(Magnitude_Pyramid{i});
            
            % weighting window for magnitudes
            gauss_weight = gauss1d(Sigmas(i)*1.5);
            gauss_weight = gauss_weight' * gauss_weight;
            s = size(gauss_weight);
            if s(1) > window_size
                m = (s(1)-1)/2+1;
                gauss_weight = gauss_weight(m-r:m+r,m-r:m+r);
            end
            
            % some quantities needed for storing information about which
            % extremum shall be eliminated or added in the end
            add_extrema_counter = 1;
            Add_Extrema = zeros(1,5);
            erase_extrema = zeros(1, length(Extrema{i}(:,1)) );
            
            % loop over all points
            for p = 1:length(Extrema{i}(:,1))
                
                % image coordinate of point
                yx = round( Extrema{i}(p,1:2));
                
                % erase, if the quantities cannot be computed (image edges)
                yx_m = yx-r; yx_p = yx+r;
                if yx_m(1) < 1 | yx_m(2) < 1 |...
                   yx_p(1) > image_size(1) | yx_p(2) > image_size(2)
                    erase_extrema(p) = 1;
                    continue;
                end
                
                % extract informatios for the point
                orient_window = Angle_Pyramid{i}((yx(1)-r):(yx(1)+r),(yx(2)-r):(yx(2)+r)); 
                mag_window = Magnitude_Pyramid{i}((yx(1)-r):(yx(1)+r),(yx(2)-r):(yx(2)+r)); 
                
                % compute a angle histogram over the window
                bins = [0:10:350]; % 10 degree-steps
                orient_window = round(orient_window);
                histogram = zeros(1,36);
                
                % determine, in which bin actual element falls
                for j = 1:length(orient_window)^2
                    [h,centers]=hist(orient_window(j), bins+5);
                    pos = find(h);
                    % weight entry with Gaussian and magnitude (instead of +1)
                    histogram(pos) = histogram(pos) + mag_window(j) * gauss_weight(j);
                end
                
                % find max in hist
                [m, m_i] = max(histogram);

                % store this orientation and additional magnitude
                Extrema{i}(p,4:5) = [bins(m_i), mean2(mag_window .* gauss_weight)];

                % fit parabola to 3 values around the peak (including the peak)
                d_theta = 0;
                % bin somewhere in the middle
                if m_i-1 >= 1 & m_i+1 <= length(bins)
                    
                    y = histogram(m_i-1:m_i+1)';
                    x = centers(m_i-1:m_i+1);
                else
                    % do this when actual bin is on hist edge
                    if m_i-1 == 0 
                        y = [0, histogram(m_i:m_i+1)]';
                        x = [-5, centers(m_i:m_i+1)]';
                    end
                    if m_i+1 > length(bins)
                        y = [histogram(m_i-1:m_i), 0]';
                        x = [centers(m_i-1:m_i), 365]';
                    end
                end
                % linear equation system
                A = [x(1)^2, x(1),1;x(2)^2, x(2),1;x(3)^2, x(3),1];
                % solve it
                a = inv(A) * y;
                % new extremum
                x_n = -a(2)/(2*a(1)); 
                % store results
                Extrema{i}(p,4) = x_n;
                d_theta = x_n - bins(m_i);
                
                % is there another peak within 80% of the max value?
                histogram(m_i) = 0;
                test = find( histogram > m*80/100);
                for k = 1:length(test)
                    % if yes: add new point with corresponding main
                    % direction
                    Add_Extrema(add_extrema_counter,:) = [yx(1), yx(2), Extrema{i}(p,3), (bins(test(k))+d_theta), mean2(mag_window .* gauss_weight)];
                    add_extrema_counter = add_extrema_counter+1;
                end
            end
            % erase points, if necessary
            erase = find(erase_extrema);
            if erase
                Extrema{i} = Extrema{i}(setdiff(1:size(Extrema{i},1),[erase]),:);
            end
            % add new points, if necessary
            if add_extrema_counter > 1
                Extrema{i} = [ Extrema{i}; Add_Extrema]; 
                clear Add_Extrema;
            end
        end
    end
end

%--------------------------------------------------------------------------
% Calculate the magnitudes and orientations of local gradients
%
% Inputs:
%   - Image_Pyramid: Cell with length(Sigmas+1): Image_Pyramid{1}: the
%     image itself; Image_Pyramid{2:length(Image_Pyramid)}: Gaussian filtered
%     images corresponding to the steps in scale space
% Outputs:
%   - Magnitude_Pyramid: Cell with length(Magnitude_Pyramid) = length(Maxima)
%       --> in each level for each pixel the local gradient magnitudes are stored
%   - Angle_Pyramid: Cell with length(Angle_Pyramid) = length(Maxima)
%       --> in each level for each pixel the local gradient orientations are stored
function [Magnitude_Pyramid, Angle_Pyramid] = ComputeOrientations(Image_pyramid, Maxima, Minima)
    
    for i = 1:length(Maxima)
        % shifted images for differences of each pixel
        dxm1 = circshift( Image_pyramid{i+1}, [0,1]);
        dxp1 = circshift( Image_pyramid{i+1}, [0,-1]);
        dym1 = circshift( Image_pyramid{i+1}, [1,0]);
        dyp1 = circshift( Image_pyramid{i+1}, [-1,0]);

        % image edges do not matter here, since we filtered edge-near keypoints out before
        Magnitude_Pyramid{i} = sqrt( (dyp1-dym1).^2 + (dxp1-dxm1).^2 );
        Angle_Pyramid{i} = atan2( dym1-dyp1, dxm1-dxp1 )*180/pi;

        % transform angles from [-pi:pi] to [0:2pi]
        neg = find(Angle_Pyramid{i} < 0);
        if neg
           Angle_Pyramid{i}(neg) = 360 + Angle_Pyramid{i}(neg);
        end
    end
end

%--------------------------------------------------------------------------
% - Refinement of keypoint positions in y,x and scale
% - Filtering out keypoint candidates by informations about contrast and
%   edge-iness using thresholds
% Inputs:
%   - Extrema: Cell with maxima or minima in each scale step-->array of size (n_points, 3)
%   - DoG_pyramid: Cell with length(DoG_pyramid) = length(Image_pyramid)-2
%     In each level the Difference of Gaussian images from neighboring
%     Gaussian blurred images are stored
%   - Sigmas: vector of lenght(Sigmas) = length(DoG_pyramid) with
%             corresponding coordinate (sigma) in scale space
%   - thresh_contrast: threshold for contrast filtering (default: 0.03)
%   - thresh_edges: threshold for edge-iness filtering (default: 10)
%   - do_shifting: boolean value
%       1: shift of rounded image coordinates
%          without filtering the keypoints (has to be done first)
%       0: filter refined keypoints using the thresholds thresh_contrast
%          and thresh_edges
% Usually this method is used twice: @ first with do_shifting=1 and then a
% second time with do_shifting = 0!
%
% Outputs:
%   - Cell with maxima or minima in each scale step-->array of size (n_points, 3)
%     do_shifting=1-->coordinates are shifted, if needed
%     do_shifting=0-->keypoints are refined + filtered
function Extrema = LocalizeContrastEdges(Extrema, DoG_pyramid, Sigmas, thresh_contrast, thresh_edges, do_shifting)
    
    size_image = size( DoG_pyramid{1} );
    % loop over scales
    for s = 2:length(DoG_pyramid)-1
        
        act_ex = Extrema{s};
        % if maxima occur in scale, then...
        if length(act_ex) >= 1
            
            n = length(act_ex(:,1));
            filter_out = zeros(n,1);
            
            % loop over all candidates in actual scale
            for p = 1:n
                
                % the following quantities are needed:
                % ip1 stands for i+1; im1 stands for i-1
                Dijk = DoG_pyramid{s}(act_ex(p,1),act_ex(p,2));
                Dijkp1 = DoG_pyramid{s+1}(act_ex(p,1),act_ex(p,2));
                Dijkm1 = DoG_pyramid{s-1}(act_ex(p,1),act_ex(p,2));
                
                Dip1jk = DoG_pyramid{s}(act_ex(p,1)+1,act_ex(p,2));
                Dim1jk = DoG_pyramid{s}(act_ex(p,1)-1,act_ex(p,2));

                Dijp1k = DoG_pyramid{s}(act_ex(p,1),act_ex(p,2)+1);
                Dijm1k = DoG_pyramid{s}(act_ex(p,1),act_ex(p,2)-1);
                
                Dijm1km1 = DoG_pyramid{s-1}(act_ex(p,1),act_ex(p,2)-1);
                Dijm1kp1 = DoG_pyramid{s+1}(act_ex(p,1),act_ex(p,2)-1);
                Dijp1km1 = DoG_pyramid{s-1}(act_ex(p,1),act_ex(p,2)+1);
                Dijp1kp1 = DoG_pyramid{s+1}(act_ex(p,1),act_ex(p,2)+1);

                Dim1jkm1 = DoG_pyramid{s-1}(act_ex(p,1)-1,act_ex(p,2));
                Dim1jkp1 = DoG_pyramid{s+1}(act_ex(p,1)-1,act_ex(p,2));
                Dip1jkm1 = DoG_pyramid{s-1}(act_ex(p,1)+1,act_ex(p,2));
                Dip1jkp1 = DoG_pyramid{s+1}(act_ex(p,1)+1,act_ex(p,2));
                
                Dim1jm1k = DoG_pyramid{s}(act_ex(p,1)-1,act_ex(p,2)-1);
                Dim1jp1k = DoG_pyramid{s}(act_ex(p,1)-1,act_ex(p,2)+1);
                Dip1jm1k = DoG_pyramid{s}(act_ex(p,1)+1,act_ex(p,2)-1);
                Dip1jp1k = DoG_pyramid{s}(act_ex(p,1)+1,act_ex(p,2)+1);
                
                % calculate second derivatives
                d_ss = (Dijkm1 - 2*Dijk + Dijkp1);
                d_yy = (Dim1jk - 2*Dijk + Dip1jk);
                d_xx = (Dijm1k - 2*Dijk + Dijp1k);

                d_sy = ( ( Dip1jkp1 - Dip1jkm1 ) - ( Dim1jkp1 - Dim1jkm1 ) )/4;
                d_sx = ( ( Dijp1kp1 - Dijp1km1 ) - ( Dijm1kp1 - Dijm1km1 ) )/4;
                d_yx = ( ( Dip1jp1k - Dip1jm1k ) - ( Dim1jp1k - Dim1jm1k ) )/4;
                
                % solve linear equation system
                A = [d_yy, d_yx, d_sy; d_yx, d_xx, d_sx; d_sy, d_sx, d_ss ];
                b = -[ (Dip1jk-Dim1jk)/2, (Dijp1k-Dijm1k)/2, (Dijkp1-Dijkm1)/2];
                dx = inv(A)*b';
                
                % finite results? if not: throw point away!
                test = find(~isfinite(dx));
                if test
                    filter_out(p) = 1;
                    continue;
                end
                
                % only do shifting, if needed
                if do_shifting
                    % test of position
                    test = find(abs(dx(1:2)) > 0.5);
                    if test
                        Extrema{s}(p,1) = Extrema{s}(p,1) + round(dx(1));
                        Extrema{s}(p,2) = Extrema{s}(p,2) + round(dx(2));                        
                        
                        % test exeeding borders in y and x
                        if Extrema{s}(p,1) < 2 | Extrema{s}(p,1) > size_image(1)-1 |...
                           Extrema{s}(p,2) < 2 | Extrema{s}(p,2) > size_image(2)-1     
                           filter_out(p) = 1;
                        end
                    end
                else
                % else filter out points, if needed
                    % contrast of extremum
                    D_n = Dijk + 0.5*[(Dip1jk-Dim1jk)/2, (Dijp1k-Dijm1k)/2, (Dijkp1-Dijkm1)/2] * dx;
                    if abs( D_n ) < thresh_contrast
                        % just store the positions to be erased
                        filter_out(p) = 1;
                    end

                    % edge-iness of extremum
                    % hessian matrix
                    H = [d_yy, d_yx; d_yx, d_xx];
                    ratio = trace(H)^2 / det(H);
                    if ratio > (thresh_edges+1)^2/thresh_edges
                        filter_out(p) = 1;
                    end

                    Extrema{s}(p,1) = Extrema{s}(p,1) + dx(1);
                    Extrema{s}(p,2) = Extrema{s}(p,2) + dx(2);
                    % store also new scale
                    Extrema{s}(p,3) = Sigmas(s) + dx(3);

                    % test exeeding borders in y
                    if Extrema{s}(p,1) < 2 | Extrema{s}(p,1) > size_image(1)-1 |...
                       Extrema{s}(p,2) < 2 | Extrema{s}(p,2) > size_image(2)-1     
                       filter_out(p) = 1;
                    end
                end
            end
            % erase positions due to edginess or contrast
            filter_out = find(filter_out);
            if filter_out
                Extrema{s} = Extrema{s}(setdiff(1:size(Extrema{s},1),[filter_out]),:);
            end
        end
    end
end

%--------------------------------------------------------------------------
% Plot the found extrema (max and mins)
% Inputs: 
%    - Maxima, Minima, Image: see above
%    - TitleOfWindow: string with title of plotted window
function PlotExtrema(Maxima, Minima, Image, TitleOfWindow)
    
    % plot image with title
    scrsz = get(0, 'ScreenSize');
    fig = figure('Position', [0,0,scrsz(3),scrsz(4)],'name', 'Results');        
    s = subplot(1,1,1); set(s, 'position', [0.1 0.1 0.8 0.8]);
    imshow(mat2gray(Image)); title(TitleOfWindow);
    
    img_size = size(Image);
    
    % plot the points of the extrema
    hold on;
    PlotM(Minima, 'g*', s, 'g-', img_size); 
    PlotM(Maxima, 'r+', s, 'r-', img_size);
    hold off;
end

%--------------------------------------------------------------------------
% Plot the extrema points (max or mins)
% Inputs: 
%    - Extrema: Maxima or Minima
%    - style: string with color and marker style
%    - SubplotHandle: pointer to subplot
%    - style: string with color and marker style
%    - line_style: string with color and marker style for orientation
%                  plotting (relevant, if extrema have orientation and magnitude information)
%    - img_size: size of plotted image
function PlotM(Extrema, style, SubplotHandle, line_style, img_size)
    
    % loop over all scales
    for i = 1:length(Extrema)
        Points = Extrema{i};
        % loop over all points in the actual scale
        if length(Points) > 0
            % plot point
            plot(Points(:,2), Points(:,1), style);
            
            s = size(Points);
            % plot also vectors, if orientation info is available
            if s(2) > 3
                or = Points(:,4);
                mag = Points(:,5);
                
                % scale magnitude for better vizualization
                mi = min(mag);
                ma = max(mag);
                mag = (mag./ (ma)) .* 0.1*min(img_size);
                
                dx = sin( or.*pi/180 ) .* abs(mag);
                dy = cos( or.*pi/180 ) .* abs(mag);
                % plot vectors
                for j = 1:length(dy(:,1))
                    plot([Points(j,2), (Points(j,2)+dy(j))], [Points(j,1), (Points(j,1)+dx(j))], line_style);
                end
            end
        end
    end
end

%--------------------------------------------------------------------------
% Identify local extrema in DoG-Space by exploring the 26 neighboring
% pixels in the DoG-pyramid
% Inputs: 
%    - DoG_pyramid: See above
%    - Sigmas: See above
% Outputs:
%   - Maxima: Cell with maxima in each scale-->array of size (n_points, 5)
%             [y,x,sigma]
%   - Minima: equivalent to Maxima
function [Maxima, Minima] = FindLocalExtrema(DoG_pyramid, Sigmas)
    
    % init result cells
    Maxima = cell(1,length(DoG_pyramid));
    Minima = cell(1,length(DoG_pyramid));
    
    % loop over scales to find candidates for local max and mins
    for scale = 2:(length(DoG_pyramid)-1)
        
        % get three DoG-images (scale, scale minus 1, scale plus 1)
        DoG_m1 = DoG_pyramid{scale-1};
        DoG_s  = DoG_pyramid{scale};
        DoG_p1 = DoG_pyramid{scale+1};
        
        % use 8 difference arrays (8-neighborhood in scale)
        n_n = sign(DoG_s - circshift(DoG_s, [-1,0]));
        n_s = sign(DoG_s - circshift(DoG_s, [1,0]));
        n_e = sign(DoG_s - circshift(DoG_s, [0,1]));
        n_w = sign(DoG_s - circshift(DoG_s, [0,-1]));
        n_ne = sign(DoG_s - circshift(DoG_s, [-1,1]));
        n_nw = sign(DoG_s - circshift(DoG_s, [-1,-1]));
        n_se = sign(DoG_s - circshift(DoG_s, [1,1]));
        n_sw = sign(DoG_s - circshift(DoG_s, [1,-1]));
        
        % we look for max and mins--> all differences for a pixel position 
        % in the above 8 arrays have to be positive or negative,
        % respectively
        for pm = -1:2:1
            candidates = (n_n == pm & n_e == pm & n_s == pm & n_w == pm &...
                         n_ne == pm & n_nw == pm & n_se == pm & n_sw == pm);
            
            % get coordinates
            [y,x] = find(candidates);
            s = size(DoG_s);

            % filter out edge pixels (descriptor mask will reach over edges)
            edges = find( ( y == 1) | y == (s(1) ) | ( x == 1) | x == (s(2) ));
            if edges
                candidates(y(edges),x(edges)) = 0;
                y = y(setdiff(1:size(y,1),[edges]),:);
                x = x(setdiff(1:size(x,1),[edges]),:);
            end
            
            % array in which max and min positions will be stored 
            test = zeros(length(y(:,1)),1);
            % loop over candidates
            for i = 1:length(y)

                % get neighboring pixels in scales sigma+/-1
                pos =[y(i),x(i)];
                chip_m1 = DoG_m1(pos(1)-1:pos(1)+1, pos(2)-1:pos(2)+1);
                chip_p1 = DoG_p1(pos(1)-1:pos(1)+1, pos(2)-1:pos(2)+1);
                chip_s = DoG_s(pos(1)-1:pos(1)+1, pos(2)-1:pos(2)+1);
                
                % test, if max or min
                if pm == 1
                    m = ( DoG_s(pos(1),pos(2)) > chip_m1 ) & ( DoG_s(pos(1),pos(2)) > chip_p1 );
                else
                    m = ( DoG_s(pos(1),pos(2)) < chip_m1 ) & ( DoG_s(pos(1),pos(2)) < chip_p1 );
                end
                % store, if test positive
                if sum(sum(m)) == 9
                    test(i) = 1;
                end
            end

            % filter non-maxima
            test = find(~test);
            if test
                y = y(setdiff(1:size(y,1),[test]),:);
                x = x(setdiff(1:size(x,1),[test]),:);
            end

            % store candidates
            if length(y) > 0
                if pm == 1
                    Maxima{scale} = [y,x];
                else
                    Minima{scale} = [y,x];
                end
            end
        end
    end
end

%--------------------------------------------------------------------------
% Create a test image with a circle
function [image, img_size] = CreateCircleImage(radius)
    
    % Groesse des Testbildes
    testSize = ceil(4*radius);
    % Matrix mit X-Koordinate im Testbild
    distMat = ones(testSize,1)*[1:testSize] - 2*radius;
    % Matrix mit Distanz zur Mitte des Testbildes
    distMat = sqrt(distMat.^2 + distMat'.^2);
    % Testbild mit Kreis berechnen
    image = (distMat <= radius);
    img_size = size(image);
end
%--------------------------------------------------------------------------
% create a test image with a chessboard
function [image, img_size] = CreateChessImage(n_pix_per_square, n_tiles_x, n_tiles_y)
    image = checkerboard(n_pix_per_square, n_tiles_x, n_tiles_y);
    img_size = size(image);
end

%--------------------------------------------------------------------------
% Calculation of the scale space images using Gaussian blurring
% Calculation of the Difference of Gaussians Pyramid using the blurred
% images
%
% Inputs: 
%    - n_int: number of intervals of the scale space
%    - image: 2d-array with the image
%    - sigma_start: smallest std. derivation of resulting scale space
%    - sigma_stop: highest std. derivation of resulting scale space
%    - plot: 1: results will be plotted; 0: results will not be plotted
%    - smooth: boolean number: 0-->no pre-smoothing of input image; 
%        1-->pre-smoothing is done (image is doubled in size before!!!)
% Outputs:
%   - DoG_pyramid: Cell of size length(Image_Pyramid)-2: 
%     In each level the Difference of Gaussian images from neighboring
%     Gaussian blurred images are stored
%   - Sigmas: See above
%   - Image_Pyramid: Cell with length(Sigmas+1): Image_Pyramid{1}: the
%     image itself; Image_Pyramid{2:length(Image_Pyramid)}: Gaussian filtered
%     images corresponding to the steps in scale space
%   - img_size: size of the image (may be changed due to smoothing!)
function [DoG_pyramid, Sigmas, Image_pyramid, img_size] = CalcDoGPyramid(n_int, image, sigma_start, sigma_stop, plot, smooth)

    % enlarge the image, if smoothing shall be done
    % store image on cell position Image_pyramid{1}
    if smooth 
        % prior upsampling
        image = resample(resample(image,2,1,5)',2,1,5)';
        % prior smoothing
        kernel = gauss1d(1.6);
        kernel = kernel' * kernel;    
        Image_pyramid{1} = fft_conv(double(image), kernel);
    else
        Image_pyramid{1} = double(image);
    end
    % imagesize may have changed...
    img_size = size(image);
    
    DoG_counter = 1;
    Ip_counter = 2;
    
    % plot result, if desired
    if plot
        scrsz = get(0, 'ScreenSize');
        figure('Position', [0,0,scrsz(3),scrsz(4)],'name', 'Scale Space results');
        d = waitbar(0, 'Processing scale space ....');
        subplot(1,2,1); imshow(Image_pyramid{1}), title('Original images');
    end
    
    % calculate factor k, which is sucessively multiplied with the standard
    % deviations
    div = sigma_stop / sigma_start;
    k = (div)^(1/n_int);
    sig = sigma_start;

    % construct a vector of sigmas corresponding to DoG-Pyramid in parallel
    for i = 1:n_int+2
        % the last sigma is not necessary to store
        if i < n_int+2
            Sigmas(i)= sig;
        end
        % provide smoothing kernel for the current sigma
        kernel = gauss1d(sig);
        kernel = kernel' * kernel;
        % multiply sigma wit k for the next iteration
        sig = sig * k;
        
        % convolve the image with the kernel in frequency domain
        Image_pyramid{Ip_counter} = fft_conv(Image_pyramid{1}, kernel);
        % plot, if desired
        if plot
            subplot(1,2,1); imshow(Image_pyramid{Ip_counter}), title('Filtered images');
        end
        
        % differences can only be calculated, when at least 2 blurred
        % images are available
        if i ~= 1
            DoG_pyramid{DoG_counter} = Image_pyramid{Ip_counter} - Image_pyramid{Ip_counter-1};
            % plot result, if desired
            if plot
                %subplot(1,2,2), imshow(imadjust(DoG_pyramid{DoG_counter})), title('DoG images');
                subplot(1,2,2), imshow(DoG_pyramid{DoG_counter}, []), title(['DoG images, sigma=' num2str(Sigmas(i-1))]);
                
%                 mini = min(DoG_pyramid{DoG_counter}(:));
%                 imwrite(imadjust(DoG_pyramid{DoG_counter}+abs(mini)), ['DoG_' num2str(Sigmas(i-1)) '.png']);
%                 imwrite(Image_pyramid{Ip_counter}, ['Gaussian_' num2str(Sigmas(i-1)) '.png']);
            end
            DoG_counter = DoG_counter + 1;
        end

        Ip_counter = Ip_counter + 1;
        if plot
            waitbar(i/(n_int+1),d);
        end
    end
end

%--------------------------------------------------------------------------
% Convolution of an image in the frequency domain
% 
% Inputs: 
%    - Image: 2d-array with the image
%    - kernel: 2d-array with the kernel
% Outputs:
%   - ConvImage: convolution result 
 function ConvImage = fft_conv(Image, kernel)
     
     % size of input quantities in image domain
     s_kernel = size(kernel);
     s_image = size(Image);
     
     % resulting sizes (padding in order to avoid circular effects on image edges)
     s1 = s_kernel(1) + s_image(1) - 1;
     s2 = s_kernel(2) + s_image(2) - 1;
     
     % f-transform of noisy image and filter
     fft_image = fft2(Image, s1, s2);
     fft_image_filter = fft2(kernel, s1, s2);
     % convolution
     fft_filtered_image = fft_image .* fft_image_filter;
     ConvImage = abs( real(ifft2(fft_filtered_image)));
     % extract the initial image area
     pad1 = ceil((s_kernel(1)-1)./2);
     pad2 = ceil((s_kernel(2)-1)./2);
     ConvImage = ConvImage(pad1+1:s_image(1)+pad1, pad2+1:s_image(2)+pad2);
 end
 

%--------------------------------------------------------------------------
% read an imagefile using a GUI
% 
% Inputs: 
%    - text: String with information for user
% Outputs:
%   - file: name of the file
%   - path: path of the file
%   - image: 2d-array of the image; if the chosen image hat more than one
%     channel, the arithmetic mean of all channels wil be computed 
%   - s: size of the image
function [file, path, image, s] = read_image(text)

    % open a dialogue to pick afile
    [file, path] = uigetfile('*.*', text);
 
    % read training image
    image = imread([path,file]);

    % determine size/dimensions of image
    s = size(image);

    % are there more than one channels?
    % --> store number of channels
    if  numberofelements(s) == 2 
        n_channels = 1;
    elseif numberofelements(s) == 3
        n_channels = s(3);
    else
        % if there are not 2 or 3 dimensions --> exit
        'Image dimensions not as expected - returning!'
        return
    end
    
    % if the image has more than one channels, simply calculate the mean of 
    % the channel values at first
    if n_channels > 1
        image = uint8( mean(image, 3) );
    end
end

%--------------------------------------------------------------------------

% Function for a self-made 1-dimensioanl gaussian filter.
% sigma = 1.0 leads to a 7-element filter
function g = gauss1d(sigma)
    % --- Calculate its size and build filter
    r = round(3*sigma); i = -r:r;
    g = exp(-i.^2/(2*sigma^2))/(sigma*sqrt(2*pi));
end
