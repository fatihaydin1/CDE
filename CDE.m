%% Class-driven Dimension Embedding (CDE)
function [ mappedX ] = CDE( X, Y, distanceMetric, numOfNeighbors, weightingFunc, TestIndices )
%   Name                        Value
%
%   'X'                            Input data
%
%   'Y'                            Class labels
%
%   'DistanceMetric'        This parameter specifies the metric that may be used to measure the distances between the data points. 
%
%   'numOfNeighbors'      The number of neighbors of a point
%
%   'weightingFunc'          This parameter specifies the weighting method. The values of the parameter are as follows: 'ln', 'lnln', 'cdf', 'adaptive'
%
%   'TestIndices'              This parameter specifies the indices of the test instances in the data set (for cross-validation).
%

    Y(TestIndices) = '<undefined>';
    [ idx, dist ] = findNeighbors( X, Y, distanceMetric, numOfNeighbors );
    [ weights ] = generateWeights( weightingFunc, dist, numOfNeighbors );
    mappedX = transform(idx, weights, Y);
end



%% Generate the weight vector
function [ weights ] = generateWeights( weightingFunc, dist, numOfNeighbors )

    % Generate the weights, according to the selected weighting function
    switch weightingFunc
        case 'ln'
            % a concave monotonically increasing function
            weights = log(2:1:(numOfNeighbors + 1));
        case 'lnln' 
            % a concave monotonically increasing function
            weights = log(log(2:1:(numOfNeighbors + 1)));
        case 'cdf'
            x1 = mean(dist(:,1:end), 1);
            weights = cdf('Normal', x1, mean(x1), std(x1));
        case 'adaptive'
            weights = normalize(mean(dist(:,1:end), 1),'range');
        otherwise
            error('%s is an unknown weighting function', weightingFunc);
    end
    weights = weights';
end



%% Find the neighbors
function [ idx, dist ] = findNeighbors( X, Y, distanceMetric, numOfNeighbors )

    if strcmp(distanceMetric, 'infogain')
        % Define the distance metric based on the information gain.
        M_IGDist = @(x,Z,g)sum(abs(x-Z).^g,2);
        gains = InformationGain(X, Y)';
        [idx, dist] = knnsearch(X, X, 'Distance',@(x,Z)M_IGDist(x,Z,gains), 'K', numOfNeighbors + 1);
    else
        [idx, dist] = knnsearch(X, X, 'Distance',distanceMetric, 'K', numOfNeighbors + 1);
    end
    
    % Remove the first column since it is the distance of a point to itself
    dist = dist(:,2:end);
    idx = idx(:,2:end);
end



%% Transform function for data points
function [ newX ] = transform( idx, weights, Y )

    % Get the list of classes
    Classes = categories(Y);
    cNum = length(Classes);

    newX = zeros(numel(Y),cNum);

    % Get all the neighbors of each point in ascending distance order
    neighbours = Y(idx);

    % Compute the new value of each point, according to the classes
    for i = 1 : cNum
        % Generate the logical-neighborhood table for the neighbors corresponding to the related class
        A = (neighbours == Classes(i));
        % Multiply the logical-neighborhood table to the weight vector
        newX(:,i) = (A*weights);
    end    
end



%% Compute the entropy of the class, i.e. Compute H(Y)
function [ H ] = H_Theorem( p )
    % Create a frequency table
    % The first column in 'T' contains the unique string values in 'p'.
    % The second is the number of instances of each value.
    % The last column contains the percentage of each value.
    T = tabulate(p);
    
    % Get a vector including the number of instances per class.
    m = cell2mat(T(:,2));
    
    % Compute probabilities per class.
    probPerClass = m ./ sum(m);
    
    % Ignore bins with 0 probability such as 0*log(0)=0.
    probPerClass = probPerClass(probPerClass > 0);
    
    % Compute Boltzmann's H Theorem.
    H = -sum(probPerClass .* log2(probPerClass));
end



%% Compute the entropy per attributes, i.e. Compute H(Y|X)
function [ eoa ] = entropyOfAttribute( X, outcome )
    eoa = 0;
    % It partitions the values in 'X' into bins, and is an array of the
    % same size as 'X' whose elements are the 'BIN' indices for the
    % corresponding elements in 'X'.
    [~, ~, BIN] = histcounts(X, 'BinMethod', 'sturges');
    
    for c = unique(BIN)'
        idx = (BIN == c);
        p = sum(idx) / length(X);
        if p > 0
            eoa = eoa + p * H_Theorem(outcome(idx));
        end
    end  
end



%% Compute the Information Gain per features, i.e. Compute IG = H(Y) - H(Y|X)
function [ gains ] = InformationGain( features, outcome )
    
    % Remove the test instances from the data set
    features(isundefined(outcome),:) = [];
    outcome(isundefined(outcome)) = [];

    % Compute the entropy of the class 
    outcomeEntropy = H_Theorem(outcome);
    
    d = size(features, 2);
    
    % Compute entropy per features.
    gains = zeros(d, 1);
    for i = 1 : d
        gains(i) = outcomeEntropy - entropyOfAttribute(features(:,i), outcome);
    end
end

