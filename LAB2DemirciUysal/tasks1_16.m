%4
% Clear workspace and close any existing figures
clear
close all

% where the first column represents years (1700-2014) and the second column represents sunspot numbers
load sunspot.txt

% Plot the sunspot data
figure(1) % Create a new figure window
plot(sunspot(:,1), sunspot(:,2), 'r-*') % Plot years on x-axis, sunspot numbers on y-axis, red line with asterisk markers

% Add axis labels and title
xlabel('Year')
ylabel('Sunspot Number')
title('Sunspot Activity (1700-2014)')

%5
% Set order of autoregressive model
n = 2;

% Determine the length of the data
L = length(sunspot);

% Create matrix P
P = [sunspot(1:L-n, 2)' ; sunspot(2:L-n+1, 2)'];

% Create matrix T
T = sunspot(n+1:L, 2)';


% Check content and size of matrices P and T
%disp('Matrix P:');
%disp(P);
disp('Size of Matrix P:');
disp(size(P));
display(P);
%disp('Matrix T:');
%disp(T);
disp('Size of T:');
disp(size(T));
display(T);
%6
% Create a new window
figure(2)

% Draw diagram
plot3(P(1,:), P(2,:), T, 'bo')

%Add axis labels and titles
xlabel('Input 1')
ylabel('Input 2')
zlabel('Output')
title('Input-Output Relationship')

%7
% Create training dataset (Pu and Tu)
Pu = P(:, 1:200);
Tu = T(1:200);

% Check the new matrices
%disp('Pu matrix:');
%disp(Pu);
disp('Size of Pu matrix:');
disp(size(Pu));

%disp('Tu matrix:');
%disp(Tu);
disp('Size of Tu matrix:');
disp(size(Tu));

%8
% Create and train the neural network
net = newlind(Pu, Tu);

%9
% Display neuron weight coefficient values
disp('Neuron weight coefficient values:');
disp('Weights:');
disp(net.IW{1});
disp('Bias:');
disp(net.b{1});

% Assign corresponding weight coefficient values to auxiliary variables
w1 = net.IW{1}(1);
w2 = net.IW{1}(2);
b = net.b{1};

%10
% Perform network simulation using training data
Tsu = sim(net, Pu);

% Compare forecasted values with known values
figure;
hold on;
plot(1702:1901, Tu, 'b', 'LineWidth', 2); % Plot known values (Tu) in blue
plot(1702:1901, Tsu, 'r--', 'LineWidth', 2); % Plot forecasted values (Tsu) in red dashed line

% Add labels and title
xlabel('Year');
ylabel('Sun Plum Activity');
title('Forecasted vs. Known Sun Plum Activity (1702-1901)');

% Add legend
legend('Known Activity', 'Forecasted Activity');

% Hold off to prevent further plots from being added to the same figure
hold off;

%11
% Perform network simulation using the remaining data set
Tsu = sim(net, P(:, 201:end));
% Extract true values for the remaining data set
Tu_remainder = T(201:end);
% Plot comparison diagram
figure;
plot(1902:2014, Tu_remainder, 'b', 'LineWidth', 2); % Plot true values (Tu_remainder) in blue
hold on;
plot(1902:2014, Tsu, 'r--', 'LineWidth', 2); % Plot forecasted values (Tsu) in red dashed line

% Add labels and title
xlabel('Year');
ylabel('Sun Plum Activity');
title('Comparison between Forecasted and True Values for Remaining Data');

% Add legend
legend('True Values', 'Forecasted Values');

% Hold off to prevent further plots from being added to the same figure
hold off;


%12
% Perform network simulation using the remaining data set to get forecasted values
Tsu = sim(net, P(:, 201:end));

% Compute forecast error vector e
e = Tu_remainder - Tsu;

% Plot error diagram
figure;
plot(e, 'b', 'LineWidth', 2);

% Add labels and title
xlabel('Data Point');
ylabel('Forecast Error');
title('Forecast Error Diagram');

% Add a horizontal line at y=0 to visualize the zero error line
hold on;
plot([1, length(e)], [0, 0], 'r--');

% Add legend
legend('Forecast Error', 'Zero Error Line');

% Hold off to prevent further plots from being added to the same figure
hold off;


%13
% Draw forecast error histogram
figure;
hist(e, 40); % Use 40 bins for the histogram

% Add labels and title
xlabel('Forecast Error');
ylabel('Frequency');
title('Forecast Error Histogram');


%14
% Calculate Mean Squared Error (MSE)
MSE = mean(e.^2);
disp(['Mean Squared Error (MSE): ', num2str(MSE)]);

% Calculate Median Absolute Deviation (MAD)
MAD = median(abs(e));
disp(['Median Absolute Deviation (MAD): ', num2str(MAD)]);






