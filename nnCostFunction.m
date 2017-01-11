function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
a1 = X;
a1 = [ones(size(a1, 1), 1) a1];

a2 = sigmoid(Theta1 * a1')';
a2 = [ones(size(a2, 1), 1) a2];

a3 = sigmoid(Theta2 * a2')'; % this is h(X)
h = a3;

Id = eye(num_labels);
yy = zeros(m, num_labels); %yy will be y_k^{(i)}
for i = 1:m
    yy(i,:) = Id(y(i),:); %Id(y(i),:) -> y_k^{(i)}
end

Theta1NoBias = Theta1(:,2:end); % Remove first column, which must not be regularized
Theta2NoBias = Theta2(:,2:end);% Remove first column, which must not be regularized
regularizationTerm = lambda / (2*m) * (trace(Theta1NoBias * Theta1NoBias') + trace(Theta2NoBias * Theta2NoBias'));

J = (1/m) * (-trace(yy * log(h')) - trace((1 - yy) * log(1 - h'))) + regularizationTerm;


delta3 = a3-yy; % rows correspond to training examples, cols correspond to a particular class, e.g. 1, 2, 3, 4, ...
delta2 = delta3 * Theta2 .* a2 .* (1-a2);
delta2 = delta2(:,2:end); % remove delta(:,0)

p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = delta2' * a1./m + p1;
Theta2_grad = delta3' * a2./m + p2;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end