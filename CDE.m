%% Class-driven Dimension Embedding (CDE)
function [ mappedX ] = CDE( X, Y, varargin )
%   Name                        Value
%
%   'X'                            Input data
%
%   'Y'                            Class labels
%
%   The struct param contains the following fields:
%   'DistanceMetric'        This parameter specifies the metric that may be used to measure the distances between the data points. 
%
%   'NumOfNeighbors'      The number of neighbors of a point
%
%   'WeightingFunc'          This parameter specifies the weighting method. The values of the parameter are as follows: 'ln', 'lnln', 'cdf', 'adaptive'
%
%   'TestIndices'              This parameter specifies the indices of the test instances in the data set (for cross-validation).
%
    
    narginchk(2, 3);
    
    if nargin == 2
        param = struct;
        param = setDefaultValues(param);
    elseif nargin == 3
        if ~isstruct(varargin{1})
            error('The third parameter must be a struct');
        end
        param = setDefaultValues(varargin{1});
    end

    Y(param.TestIndices) = '<undefined>';
    [ idx, dist ] = findNeighbors( X, Y, param );
    [ weights ] = generateWeights( param.WeightingFunc, dist, param.NumOfNeighbors );
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
function [ idx, dist ] = findNeighbors( X, Y, param )

    if strcmp(param.DistanceMetric, 'infogain')
        % Define the distance metric based on the information gain.
        M_IGDist = @(x,Z,g)sum(abs(x-Z).^g,2);
        gains = InformationGain(X, Y)';
        [idx, dist] = knnsearch(X, X, 'Distance',@(x,Z)M_IGDist(x,Z,gains), 'K', param.NumOfNeighbors + 1);
    else
        [idx, dist] = knnsearch(X, X, 'Distance',param.DistanceMetric, 'K', param.NumOfNeighbors + 1);
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



%%
function [ param ] = setDefaultValues( param )

    field = {'DistanceMetric', 'NumOfNeighbors', 'WeightingFunc', 'TestIndices'};
    TF = isfield(param, field);
    
    if TF(1) == 0
        param.DistanceMetric = 'cityblock';
    end
    
    if TF(2) == 0
        param.NumOfNeighbors = 5;
    end
    
    if TF(3) == 0
        param.WeightingFunc = 'adaptive';
    end
    
    if TF(4) == 0
        param.TestIndices = [];
    end
end


