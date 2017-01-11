function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
% Note: The first column of W corresponds to the parameters for the bias unit

W = zeros(L_out, 1 + L_in);

% Randomly initialize the weights to small values so that we break the symmetry while
%               training the neural network.
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;


% =========================================================================

end
