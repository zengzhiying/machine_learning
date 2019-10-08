function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

min_error = 1000;
C_sigma = [0.01 0.03 0.1 0.3 1 3 10 30;];
for i_c = 1:length(C_sigma),
    for i_sigma = 1:length(C_sigma),
        test_c = C_sigma(i_c);
        test_sigma = C_sigma(i_sigma);
        model = svmTrain(X, y, test_c, @(x1, x2) gaussianKernel(x1, x2, test_sigma));
        predictions = svmPredict(model, Xval);
        mean_error = mean(double(predictions ~= yval));
        if mean_error < min_error,
            min_error = mean_error;
            sigma = test_sigma;
            C = test_c;
        end;
    end;
end;





% =========================================================================

end
