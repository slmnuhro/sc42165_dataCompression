%% SC42165/EE4740 Data Compression
% Convex Optimization-based CS Under Measurement Saturation
% Student 1: Sven Rutgers - 4600150
% Student 2: Melis Orhun - 4912071
%
% NOTE: The algorithm requires CVX library to be downloaded from the 
% following link: https://cvxr.com/cvx/
%
% ======================================================================= %

clc
clear
close all

% ======================================================================= %

% Set interpreter and layout options
set(groot,'defaulttextinterpreter','latex');  
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaultLegendInterpreter','latex'); 
set(groot,'defaultLineLineWidth',1)
set(groot,'defaultAxesFontSize',11)
set(groot,'defaultAxesFontWeight',"normal")

% ======================================================================= %

x_train_tbl = readtable('mnist_train.csv','ReadRowNames',true);
x_train = x_train_tbl{:,:}'-128;
x_test_tbl = readtable('mnist_test.csv','ReadRowNames',true);
x_test = x_test_tbl{:,:}'-128;

% ======================================================================= %

% Initialize compressed sensing parameters
x_data = x_test(:, 1:100); % Set signal of interest
n = length(x_data(:,1)); % Length of the signal
m = 350;  % Number of measurements
k = 50;   % Sparsity level

set_size = length(x_data(1,:));

% Generate a random sensing matrix

A = (rand(m, n) < 0.01);
A = A / sqrt(m); 

% Generate y_in = A*x + v where v is random noise
y_in = zeros(m,set_size);
v = zeros(m,set_size);

snr = 40; %dB

for i = 1:1:set_size

    signal = A * x_data(:,i);
    % Calculate the power of the original signal
    signal_power = mean(signal.^2);
    
    % Calculate the desired noise power based on the SNR
    snr_power = 10^(snr / 10);  % Convert SNR from dB to linear scale
    noise_power = signal_power / snr_power;
    
    % Generate noise with the calculated noise power
    v(:,i) = sqrt(noise_power) * randn(m,1);

    % Generate noise
    %v(:,i) = 0.01 * randi([-128,127], m, 1);

    % Linear compressed sensing measurements
    y_in(:,i) = A * x_data(:,i) + v(:,i);

end

% Define a saturation function
mean_sparse = mean(mean(y_in));
std_sparse = std(y_in(:));
saturation_threshold = abs(mean_sparse) + std_sparse;
Sat = @(y) min(max(y, -saturation_threshold), saturation_threshold);

% Saturated measurements
y_out = Sat(y_in);

% Define NMSE function
mse = @(x,y) mean((x(:) - y(:)).^2);
nmse = @(x,y) mse(x,y)/mean(x(:).^2);

% Recovery 1: Using y_in = Ax + v
x_recovery1 = zeros(n,set_size);

for i = 1:1:set_size
    cvx_begin quiet
        variable x1(n)
        minimize(norm(x1, 1))
        subject to
            norm(A * x1 - y_in(:,i), 2) <= norm(v(:,i), 2)
            x1 <= 127
            -x1 <= 128
    cvx_end
    x_recovery1(:,i) = x1;
    disp(i)
end

nmse_recovery1 = nmse(x_data,x_recovery1);

idx = randi([1,100],1,3);
% x_recovery1 = clip(x_recovery1+128, 0, 255);
x_recovery1 = x_recovery1+128;

figure;
subplot(3, 2, 1); imshow(reshape((x_data(:,idx(1))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 2); imshow(reshape((x_recovery1(:,idx(1)))./255, [28, 28])'); title('Recovered Image 1');
subplot(3, 2, 3); imshow(reshape((x_data(:,idx(2))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 4); imshow(reshape((x_recovery1(:,idx(2)))./255, [28, 28])'); title('Recovered Image 2');
subplot(3, 2, 5); imshow(reshape((x_data(:,idx(3))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 6); imshow(reshape((x_recovery1(:,idx(3)))./255, [28, 28])'); title('Recovered Image 3');

%% Recovery 2: Using y_out = Ax + v (ignoring clipping effects)
x_recovery2 = zeros(n,set_size);

for i = 1:1:set_size
    cvx_begin
        variable x2(n)
        minimize(norm(x2, 1))
        subject to
            norm(A * x2 - y_out(:,i), 2) <= norm(v(:,i), 2)
    cvx_end
    x_recovery2(:,i) = x2;
    disp(i)

end

nmse_recovery2 = nmse(x_data,x_recovery2);
x_recovery2 = x_recovery2+128;

figure;
subplot(3, 2, 1); imshow(reshape((x_data(:,idx(1))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 2); imshow(reshape((x_recovery2(:,idx(1)))./255, [28, 28])'); title('Recovered Image 1');
subplot(3, 2, 3); imshow(reshape((x_data(:,idx(2))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 4); imshow(reshape((x_recovery2(:,idx(2)))./255, [28, 28])'); title('Recovered Image 2');
subplot(3, 2, 5); imshow(reshape((x_data(:,idx(3))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 6); imshow(reshape((x_recovery2(:,idx(3)))./255, [28, 28])'); title('Recovered Image 3');

%% Recovery 3: Using only non-clipped measurements
x_recovery3 = zeros(n,set_size);

for i = 1:1:set_size
    cvx_begin quiet

        non_clipped_indices = find(abs(y_out(:,i)) < saturation_threshold);

        variable x3(n)
        minimize(norm(x3, 1))
        subject to
          norm(A(non_clipped_indices, :) * x3 - y_out(non_clipped_indices,i), 2) <= norm(v(non_clipped_indices,i), 2)
          x3 <= 127
          -x3 <= 128
    cvx_end
    x_recovery3(:,i) = x3;
    disp(i)

end

nmse_recovery3 = nmse(x_data,x_recovery3);
x_recovery3 = x_recovery3+128;

figure;
subplot(3, 2, 1); imshow(reshape((x_data(:,idx(1))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 2); imshow(reshape((x_recovery3(:,idx(1)))./255, [28, 28])'); title('Recovered Image 1');
subplot(3, 2, 3); imshow(reshape((x_data(:,idx(2))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 4); imshow(reshape((x_recovery3(:,idx(2)))./255, [28, 28])'); title('Recovered Image 2');
subplot(3, 2, 5); imshow(reshape((x_data(:,idx(3))+128)./255, [28, 28])'); title('Original Image');
subplot(3, 2, 6); imshow(reshape((x_recovery3(:,idx(3)))./255, [28, 28])'); title('Recovered Image 3');

%% Varying parameters

% Varying m

sat = 50;
snr = 40;
m_ = 300:25:400;
nmse_m = zeros(1,length(m_));

for i = 1:length(m_)

    m = m_(i);

    x_recovery_m = zeros(n,set_size);

    for j = 1:1:set_size

        [A,y_out,v] = create_dataset(m,snr,sat,x_data(:,j),n);

        cvx_begin quiet
    
            non_clipped_indices = find(abs(y_out) < sat);
    
            variable x3(n)
            minimize(norm(x3, 1))
            subject to
              norm(A(non_clipped_indices, :) * x3 - y_out(non_clipped_indices), 2) <= norm(v(non_clipped_indices), 2)
              x3 <= 127
              -x3 <= 128
        cvx_end
        x_recovery_m(:,j) = x3;
    end
    disp(m)
    nmse_m(i) = nmse(x_data,x_recovery_m);

end

%%
figure;
plot(m_,nmse_m)
title("NMSE vs. M (Number of Measurements)")
xlabel("m")
ylabel("NMSE")
grid minor

%%
% Varying snr

sat = 50;
snr_ = 30:2.5:40;
m = 350;
nmse_snr = zeros(1,length(snr_));

for i = 1:length(snr_)

    snr = snr_(i);

    x_recovery_snr = zeros(n,set_size);

    for j = 1:1:set_size

        [A,y_out,v] = create_dataset(m,snr,sat,x_data(:,j),n);

        cvx_begin quiet
    
            non_clipped_indices = find(abs(y_out) < sat);
    
            variable x3(n)
            minimize(norm(x3, 1))
            subject to
              norm(A(non_clipped_indices, :) * x3 - y_out(non_clipped_indices), 2) <= norm(v(non_clipped_indices), 2)
              x3 <= 127
              -x3 <= 128

        cvx_end

        x_recovery_snr(:,j) = x3;

    end

    disp(snr)
    nmse_snr(i) = nmse(x_data,x_recovery_snr);

end

%%
figure;
plot(snr_,nmse_snr)
title("NMSE vs. Signal-to-Noise Ratio")
xlabel("SNR")
ylabel("NMSE")
grid minor
%%
% Varying sat

sat_ = 50:5:70;
snr = 40;
m = 350;
nmse_sat = zeros(1,length(sat_));

for i = 1:length(sat_)

    sat = sat_(i);

    x_recovery_sat = zeros(n,set_size);

    for j = 1:1:set_size

        [A,y_out,v] = create_dataset(m,snr,sat,x_data(:,j),n);

        cvx_begin quiet
    
            non_clipped_indices = find(abs(y_out) < sat);
    
            variable x3(n)
            minimize(norm(x3, 1))
            subject to
              norm(A(non_clipped_indices, :) * x3 - y_out(non_clipped_indices), 2) <= norm(v(non_clipped_indices), 2)
              x3 <= 127
              -x3 <= 128

        cvx_end

        x_recovery_sat(:,j) = x3;

    end

    disp(sat)
    nmse_sat(i) = nmse(x_data,x_recovery_sat);

end

%%
figure;
plot(sat_,nmse_sat)
title("NMSE vs. Saturation Point $\alpha$")
xlabel("$\alpha$")
ylabel("NMSE")
grid minor

%%


% Function to create dataset for iterations
function [A,y_out,v] = create_dataset(m_,snr_,sat_,x_,n)

    % Generate a random sensing matrix
    A = (rand(m_, n) < 0.01);
    A = A / sqrt(m_); 
    
    % Generate y_in = A*x + v where v is random noise
    signal = A * x_;

    % Calculate the power of the original signal
    signal_power = mean(signal.^2);
    
    % Calculate the desired noise power based on the SNR
    snr_power = 10^(snr_ / 10);  % Convert SNR from dB to linear scale
    noise_power = signal_power / snr_power;
    
    % Generate noise with the calculated noise power
    v = sqrt(noise_power) * rand(m_,1);
    
    % Linear compressed sensing measurements
    y_in = A * x_ + v;

    % Define a saturation function
    saturation_threshold = sat_;
    Sat = @(y) min(max(y, -saturation_threshold), saturation_threshold);
    
    % Saturated measurements
    y_out = Sat(y_in);

end
