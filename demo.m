%% Demo for the CDE algorithm
clc;

% Select a data set
dataset = load('ovariancancer');
fns = fieldnames(dataset);
[ X, Y ] = divideTable( dataset.(fns{1}) );

% Set the parameters of CDE
DistanceMetric = 'cityblock';
numOfNeighbors = 5;
weightingFunc = 'adaptive';

% Perform an experiment with 10-fold cross-validation
Accuracy = demo1( X, Y, 'CART', DistanceMetric, numOfNeighbors, weightingFunc );

% Find the embeddings of input data
mappedX = demo2( X, Y, DistanceMetric, numOfNeighbors, weightingFunc );

clear dataset;
clear fns;
clear DistanceMetric;
clear numOfNeighbors;
clear weightingFunc;



%% Perform an experiment with 10-fold cross-validation
function [ acc ] = demo1( X, Y, classifier, DistanceMetric, numOfNeighbors, weightingFunc )

    predictions = repmat(Y, 1, 2);
    indices = crossvalind('Kfold', Y, 10);
    
    for i = 1:10
        fprintf('%d',i);
        test = (indices == i);
        train = ~test;
                
        mappedX = CDE(X, Y, DistanceMetric, numOfNeighbors, weightingFunc, test);
        trainX = mappedX(train,:);
        testX = mappedX(test,:);
        trainY = Y(train,:);

        switch classifier
            case 'CART'
                Mdl = fitctree(trainX, trainY);
            case 'KNN'
                Mdl = fitcknn(trainX, trainY);
            case 'NB'
                % "normal", "mn", "kernel", "mvmn".
                Mdl = fitcnb(trainX, trainY, 'DistributionNames', 'normal');
            case 'SVM'
                % "linear", "gaussian", "rbf", "polynomial"
                t = templateSVM('Standardize', true, 'KernelFunction', 'linear');
                Mdl = fitcecoc(trainX, trainY, 'Learners', t);
        end
        % Predict the output of an identified model
        predictions(test, 2) = predict(Mdl, testX);
    end
    acc = sum(predictions(:,1) == predictions(:,2))*100/length(Y);
end



%% Find the embeddings of input data
function [ mappedX ] = demo2( X, Y, DistanceMetric, numOfNeighbors, weightingFunc )

    mappedX = CDE(X, Y, DistanceMetric, numOfNeighbors, weightingFunc, []);
end



%% Separate the dataset into the input matrix and the output vector
function [ X, Y ] = divideTable( DATASET )

    if istable(DATASET)
        X = table2array(DATASET(:,1:end-1));        
        Y = categorical(DATASET.Class);
    else
        error('The parameter must be a table, not a %s.', class(DATASET));
    end
end

