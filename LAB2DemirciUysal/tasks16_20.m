% Load sunspot data and plot
clear
close all
load sunspot.txt
figure(1)
plot(sunspot(:,1), sunspot(:,2), 'r-*')
xlabel('Year')
ylabel('Sunspot Number')
title('Sunspot Activity (1700-2014)')

% Set order of autoregressive model
n = 2;
L = length(sunspot);
P = [sunspot(1:L-n, 2)' ; sunspot(2:L-n+1, 2)'];
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
lr = maxlinlr(Pu, 'bias');
display(lr);
net = newlin(minmax(Pu), 1, 0,lr);
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

disp(['Goal Parameter: ', num2str(net.trainParam.goal)]);
disp(['Epoch Parameter: ', num2str(net.trainParam.epochs)]);
