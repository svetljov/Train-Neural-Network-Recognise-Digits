%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Training Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 2: Initializing Parameters ================
%  We will implement a three-layer neural network that classifies digits.
%  We will start by implementing a function to initialize the weights of
%  the neural network (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Part 3: Training NN ===================
%  "fminunc" is able to train our cost functions efficiently as long as
%  we provide it with the gradient computations.

fprintf('\nTraining Neural Network... \n')

%  Change the MaxIter to a larger value to see how more training helps.
options = optimset('MaxIter', 300);

%  Try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 4: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. We will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%% ============= Part 5: Recognise user's handwriting ==============
%  Here you can supply your own 20x20 pixel BMP image containing a written
%  number by you. The number must be in black on white font.

numberBMP = 1 - imread('number.bmp'); % we substract from 1 to obtain negative image
numberBMP = reshape(numberBMP,[1, input_layer_size]);

% save('weightsTheta.mat','Theta1','Theta2');
% load('weightsTheta1.matt'); % load variable Theta1

displayData(numberBMP);
disp(predict(Theta1, Theta2, numberBMP));
