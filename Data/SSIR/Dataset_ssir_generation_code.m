clc;
clear all;
cd \path\;
rng('default')
N = 2000; beta =0.5; gamma = 0.5;
s_train = [];
i_train = [];
y_train = [];
n = 60;
r = 50;
for i = 1:1:n
    S_start = randi([1200 1800]);
    I_start = randi([20 200]);
    R = N - I_start - S_start;
    for j = 1:1:r
        y_sir = sir_function(beta,gamma,S_start,I_start,R,N);
        y_train(end+1) = S_start - y_sir;
        s_train(end+1) = S_start;
        i_train(end+1) = I_start;
    end
end
A = zeros(n*r,3);
A(:,1) = s_train;
A(:,2) = i_train;
A(:,3) = y_train;
T = array2table(A);
T.Properties.VariableNames(1:3) = {'S_start','I_start','Y_train'};
writetable(T,'Stochasticsir60X50.csv');
