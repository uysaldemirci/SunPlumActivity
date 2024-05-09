% Set order of autoregressive model
n = 6; % Update the order

% Redefine P and T matrices
L = length(sunspot);
P = zeros(n, L-n);
for i = 1:n
    P(i, :) = sunspot(i:L-n+i-1, 2)';
end
T = sunspot(n+1:L, 2)';

% Create a new window and draw diagram
figure(2)
plot3(P(1,:), P(2,:), T, 'bo')
xlabel('Input 1')
ylabel('Input 2')
zlabel('Output')
title('Input-Output Relationship')

% Create training dataset
Pu = P(:, 1:200);
Tu = T(1:200);

% Create and train the neural network using iterative method
lr = maxlinlr(P, 'bias');
net = newlin(minmax(Pu), 1, 0, lr); % Input delay = 0, Learning rate = 0.01
net.trainParam.goal = 100;
net.trainParam.epochs = 1000;
net = train(net, Pu, Tu);

% Display new weight coefficients
disp('New weight coefficients:');
disp('Weights:');
disp(net.IW{1});
disp('Bias:');
disp(net.b{1});

% Perform network simulation using the remaining data set
Tsu = sim(net, P(:, 201:end));
Tu_remainder = T(201:end);
e = Tu_remainder - Tsu;
MSE = mean(e.^2);
MAD = median(abs(e));
disp(['Mean Squared Error (MSE): ', num2str(MSE)]);
disp(['Median Absolute Deviation (MAD): ', num2str(MAD)]);
