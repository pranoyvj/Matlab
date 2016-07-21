
% Implementation of a split-based EM-algorithm
%---------------------------------------------
%
% Inputs:
%   trainVect: array of size n_featurevects x n_dimensions
%   n_comp: number of desired components in the model
%
% Outputs: 
%   model: structure with estimated model
%   model.weight: (n_components x 1) vector with weight for each component
%   model.mean: (n_components x n_dims=3) matrix with mean vectors for each
%                component --> mean of vector i: model.mean(i,:)
%   model.covar: (n_components x n_dims=3 x n_dims=3) matrix with
%                covariance matrices for each component
%                squeeze(model.covar(i,:,:)) returns the i'th covar matrix
%
% main function of the EM-algorithm
function model = LearnGaussMixModel(trainVect, n_comp)

    % initialization of the model using a structure
    % at the starting point the algorithm will always be initialized with 
    % one cluster 
    [n_vect, n_dims] = size(trainVect);
    
    model.weight(1,:)=1;
    model.mean=zeros(1,n_dims);%[0,0];%changed
    model.covar(1,:,:)=diag(ones(1,n_dims));%[1 0; 0 1];
    disp(model.covar(1,:,:));
    % threshold for stopping the iteration
    eps=10^-6;
    
    % loop over the desired number of components
    for i=1:n_comp
        
        % the first overall model probability is -infinity
        LastPX=-inf;
        
        % calculate the logarithmic overall model probability
        LnTotalProb = CalcLnTotalProb(model, trainVect);
        
        % while the threshold is bigger than the difference of the overall
        % probabilities from this and the last iteration...
        while (LnTotalProb-LastPX>eps)
            LastPX = LnTotalProb;
            
            % E-step:
            % compute for each feature vector the probabilities for each
            % component
            LnCompProb = GmmEStep(model, trainVect);
            
            % M-step:
            % Maximize the model by reestimating the model parameters using
            % the probabilities of E-step
            model = GmmMStep(model, trainVect, LnCompProb);
            
            % again compute the overall probability of the model
            % since EM always converges, this value is always higher than
            % in the last iteration
            LnTotalProb = CalcLnTotalProb(model, trainVect);      
        end
        
        % clear current figure window
        clf
        % plot the estimated GMM
        PlotGMM(model,trainVect);
        % flush
        drawnow;
        
        % find a component to split into two and init them
        % but only, if the desired number n_comp is not reached
        if i < n_comp
            model = InitNewComponent(model, trainVect);
        end
    end
end


%--------------------------------------------------------------------------
% logarithmic probability of all vectors for all components
function LnVectorProb = CalcLnVectorProb(model, trainVect)

    % number of feature vectors
    n_vect = size(trainVect);
    n_vect = n_vect(1);
    
    % number of components (clusters)
    n_comp = numel(model.weight);

    % loop over all components
    for j=1:n_comp
        
        % loop over all feature vectors (i.e., pixels)
        for i=1:n_vect
            
            % calculate the corresonding logarithmic probability for the
            % current feature vector i and the component j
            LnVectorProb(j,i) = log(model.weight(j))-1/2*((log(det(squeeze(model.covar(j,:,:)))))+((trainVect(i,:)-model.mean(j,:))*inv(squeeze(model.covar(j,:,:)))*(trainVect(i,:)-model.mean(j,:))'));
        end
    end
end

%--------------------------------------------------------------------------
% E-Step:
% calculation of the probabilities of each feature vector wrt all existing
% components using the current model parameters
function LnCompProb = GmmEStep(model, trainVect)

    % current number of components
    n_comp = numel(model.weight);
    
    % logarithmic probability of all vectors for all components
    % this is the enumerator of p(y=c|x,omega)
    LnVectorProb = CalcLnVectorProb(model, trainVect);
    
    % now, we have to divide these values through the sum of all possible
    % probabilities for each feature vector --> normalization

%     % first way:
%     % take exp of values --> summarize --> log of these sums
%     s = log( sum( exp( LnVectorProb ), 1 ));
%     s = repmat(s, n_comp, 1);
%     
%     % divide values by difference lnFeatureProb - s 
%     LnCompProb = LnVectorProb - s;
    
    % better way:
    % the above approach of normalization can lead to numerical problems
    % do normalization here!
    % 1) get the maximum probability for each feature vector
    max_LnVectorProb = max(LnVectorProb,[],1);
    % 2) resize array according to number of components
    scaling_factors = repmat(max_LnVectorProb, n_comp, 1);
    % 3) subtract scaling_factors from LnVectorProb
    % 4) take exp of the result
    % 5) summarize the n_comp values for each feature vector
    % 6) take the logarithm of the sums
    % 7) add the maximum to the result of 6 --> scaling denominator for the
    %    probabilities in LnVectorProb
    denominator = max_LnVectorProb + log(sum(exp(LnVectorProb - scaling_factors),1));
    
    % the result "denominator" is a vector --> reshape to apply the
    % division to all feature vector values
    denominator = repmat(denominator, n_comp, 1);
    
    % computatiion of the wanted probabilities LnCompProb
    % --> division of numbers is equal to difference of their log values
    LnCompProb = LnVectorProb - denominator;
end

%--------------------------------------------------------------------------
% M-Step:
% Estimation of new model parameters according the calculated probabilities
% of the E-Step
function model = GmmMStep(model, trainVect, LnCompProb)
    
    % we now need the real values (probabilities in the range [0,...,1]) 
    % instead of the logarithmic ones --> apply exp function
    CompProb = exp(LnCompProb);
    
    % derive number of components, overall points and dimensions
    n_comp = numel(model.weight);
    n_points = size(trainVect, 1);
    n_dims = size(trainVect, 2);
    
    % sum of points in each component
    n_points_per_comp = sum(CompProb, 2);
    
    % computation of weights for each component
    % this is the percentual coverage of each component wrt all points
    model.weight = n_points_per_comp/n_points;
    
    % computation of new model parameters: mean vectors and covariance
    % matrices for each component
    
    % loop over all components
    for i=1:n_comp    
        
        % array for summation for mean value calculation
        sum_for_mean = zeros(1,n_dims);
        
        % loop over all points (feature vectors!)
        for j=1:n_points        
            % summarize values
            sum_for_mean = (sum_for_mean + trainVect(j,:)*CompProb(i,j));                
        end
        % divide sum through corresponding number of point in the component
        model.mean(i,:) = sum_for_mean / n_points_per_comp(i,1);

        % also compue covariance matrix in loop over all components

        % initialize sum for covar-matrix of corresonding component
        sum_for_covar = zeros(n_dims, n_dims);    
        
        % again, loop over all feature vectors
        for j=1:n_points
            % summarize values of weighted dyadic product
            d = (trainVect(j,:)-model.mean(i,:));
            sum_for_covar = sum_for_covar + (d' * d) * CompProb(i,j);
        end
        % also divide trough the number of point in that component
        model.covar(i,:,:) = ((1/n_points_per_comp(i,1))*sum_for_covar);    
    end
end

%--------------------------------------------------------------------------
% calculation of the global probability given the current model and the
% feature vectors
function LnTotalProb = CalcLnTotalProb(model, trainVect)
    
    % get the current number of components in the model
    n_comp = numel(model.weight);

    % logarithmic probability for all vectors in all components
    LnVectorProb = CalcLnVectorProb(model, trainVect);
        
%     % the log of a sum cannot easity be computed from single log-values!
%     % so for this step we have to use the exp-function and afterwards take
%     % the log of the sum! (log of a product-->sum log values, but there is 
%     % no such rule for log of a sum!)
%     s = sum(exp(LnVectorProb),1);
%     LnTotalProb = sum(log(s));

    % the result abuve could be wrong tue to very small values after exp...
    % to be safe, we can compute LnTotalProb using a scaling factor:
    %---------------------------------------------------------------------
    % use scaling factor c = max of log values
    % wrt a feature vector --> one scale factor for each feature vctor

    % max probability for each feature vector
    % [ the maximum probability tells us, to which cluster each vector
    % belongs]
    % this value is used for scaling the probabilities in order to avoid
    % numerical problems for computation of the sum
    max_LnVectorProb = max( LnVectorProb,[],1 );
    
    % resize this array to size of LnTotalProb
    scaling_factors = repmat(max_LnVectorProb, n_comp, 1);
    
    % scaling of logarithmic probabilities before using exp in order to
    %  avoid numerical problems:
    % 1) subtract scaling_factors from LnVectorProb (scaling)
    % 2) take exp of the result (should be no problem after scaling)
    % 3) summarize the n_comp values for each feature vector (as desired)
    % 4) take the logarithm of the sums
    % 5) add the maximum to the result of 4 ("unscaling")
    LnVectorProb_new = max_LnVectorProb + log(sum(exp(LnVectorProb - scaling_factors),1));
 
    % sum all log values to get global model probability
    LnTotalProb = sum(LnVectorProb_new);
end




%--------------------------------------------------------------------------
% function for plotting a result of EM-estimation
% 
% Inputs:
%   model: Gaussian Mixture Model Parametes (structure)
%   features: feature vectors
% 
%  - plots a mixture model in a 3dim- plot (feature vector has to be 3-dimensional!)
%  - Feature vectors are plotted as green dots
%  - Means of components: red circles
%  - Covariance matrices: plotted using the three main axes of the
%    ellipsoid
function PlotGMM(model, trainVect)

    % number of components in the model
    n_comp = numel(model.weight);
    n_dims = size(trainVect, 2);
    
    hold on;
    
    % extract the 
    
    
    % plot feature vector points
    plot(trainVect(:,1),trainVect(:,2), 'g.','MarkerSize',7);

    % plot elements of the estimated components:
    for i=1:n_comp
        
        % eigenvektor / eigenwert decomposition
        [eVec,eVal] = eig(squeeze(model.covar(i,:,:)));

        % plotting of mean values
        mean = squeeze(model.mean(i,:));
        plot(mean(1),mean(2),'ro');

        % derivation and plotting of the three main axes of the cvariance
        % matrices
        for i=1:n_dims
            devVec = (sqrt(eVal(i,i)) * eVec(:,i))*[-1,1];
            plot(mean(1) + devVec(1,:), mean(2) + devVec(2,:),'b');
        end
    end

    % rotate 3D view and setting of title
    hold off;
    %view([19,25]);
    grid('on');
    title(['Gaussian Mixture Model (',num2str(n_comp),' components)']);
end

% ------------------------------------------------------------------------
% function for splitting a single component and initialization 2 new ones
%
% Inputs:
%
%   model: GMM parameters
%   features: feature vectors
%   features(j,:): feature vector of a single pixel
%   newModel: updated model with new initialized component
%
% adds a new component (cluster) to the current model
%
% --> analyzes the current model and identifies the weakest component
% --> weak: doesn't fit well to the corresponding feature vectors
% the weakest component will be splitted into two new ones
function NewModel = InitNewComponent(model, trainVect)

    % number of components
    n_comp = numel(model.weight);

    % Number of dimensions (shall be three here!)
    n_dims = size(trainVect, 2);

    % the biggest component will be splitted to get a balanced size of
    % components --> not the optimal criterium!!!!
    % size corresponds to weights...
    [ignore, splitComp] = max(model.weight);

    % calculate new weight vector, mean and covariance
    newWeight = zeros(n_comp+1,1);
    newMean = zeros(n_comp+1,n_dims);
    newCovar = zeros(n_comp+1,n_dims,n_dims);

    % copy old values into new arrays
    newWeight(1:n_comp) = model.weight;
    newMean(1:n_comp,:) = model.mean;
    newCovar(1:n_comp,:,:) = model.covar;

    % Component splitComp will be splitted along its dominant axis
    [eVec,eVal] = eig(squeeze(newCovar(splitComp,:,:)));
    [ignore, majAxis] = max(diag(eVal));
    devVec = sqrt(eVal(majAxis,majAxis)) * eVec(:,majAxis)';

    % initialize new component
    % --> half of the points --> half weight
    newWeight(n_comp+1) = 0.5*newWeight(splitComp);
    % shift new mean to half of length along dominant axis
    newMean(n_comp+1,:) = newMean(splitComp,:) - 0.5*devVec;
    % make covariance a little bit smaller
    newCovar(n_comp+1,:,:) = newCovar(splitComp,:,:) / 4.0;

    % update also the (old) splitted component
    % also half of the points
    newWeight(splitComp) = newWeight(n_comp+1);
    % shift comonent center to other direction along dominant axis
    newMean(splitComp,:) = newMean(n_comp+1,:) + devVec;
    % take same smaller covariance matrix
    newCovar(splitComp,:,:) = newCovar(n_comp+1,:,:);

    % store new parameters in model
    NewModel.weight = newWeight;
    NewModel.mean = newMean;
    NewModel.covar = newCovar;
end
