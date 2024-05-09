% Define n values from 2 to 30
n_values = 2:30;

% Define arrays to store results
MSE_values = zeros(size(n_values));
MAD_values = zeros(size(n_values));
w1_values = zeros(size(n_values));
w2_values = zeros(size(n_values));
bias_values = zeros(size(n_values));

% Loop over different values of n
for i = 1:length(n_values)
    % Set order of autoregressive model
    n = n_values(i);
    
    % Redefine P and T matrices
    L = length(sunspot);
    P = zeros(n, L-n);
    for j = 1:n
        P(j, :) = sunspot(j:L-n+j-1, 2)';
    end
    T = sunspot(n+1:L, 2)';
    
    % Create training dataset
    Pu = P(:, 1:200);
    Tu = T(1:200);
    
    % Create and train the neural network using iterative method
    lr = maxlinlr(P, 'bias');
    net = newlin(minmax(Pu), 1, 0, lr); % Input delay = 0, Learning rate = 0.01
    net.trainParam.goal = 100;
    net.trainParam.epochs = 1000;
    net = train(net, Pu, Tu);
    
    % Perform network simulation using the remaining data set
    Tsu = sim(net, P(:, 201:end));
    Tu_remainder = T(201:end);
    e = Tu_remainder - Tsu;
    
    % Calculate MSE and MAD
    MSE_values(i) = mean(e.^2);
    MAD_values(i) = median(abs(e));
    
    % Store weight coefficients and bias
    w1_values(i) = net.IW{1}(1);
    w2_values(i) = net.IW{1}(2);
    bias_values(i) = net.b{1};
end

% Plot MSE and MAD vs n
figure;
subplot(2, 1, 1);
plot(n_values, MSE_values, 'b-*');
xlabel('n');
ylabel('Mean Squared Error (MSE)');
title('MSE vs n');

subplot(2, 1, 2);
plot(n_values, MAD_values, 'r-*');
xlabel('n');
ylabel('Median Absolute Deviation (MAD)');
title('MAD vs n');

% Plot weight coefficients and bias vs n
figure;
subplot(3, 1, 1);
plot(n_values, w1_values, 'g-*');
xlabel('n');
ylabel('Weight w1');
title('Weight w1 vs n');

subplot(3, 1, 2);
plot(n_values, w2_values, 'm-*');
xlabel('n');
ylabel('Weight w2');
title('Weight w2 vs n');

subplot(3, 1, 3);
plot(n_values, bias_values, 'k-*');
xlabel('n');
ylabel('Bias');
title('Bias vs n');
