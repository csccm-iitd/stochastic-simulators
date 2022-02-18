clc;
clear all;
cd \path\;
rng('default')
y_result = [];
S_start = 1500; I_start = 80; N = 2000; R = N - I_start - S_start; beta =0.5; gamma = 0.5;
for i = 1:1:2000
    y_sir = sir_function(beta,gamma,S_start,I_start,R,N);
    y_result(end +1) = S_start - y_sir;
end
T = array2table(y_result');
T.Properties.VariableNames(1) = {'Send_minus_Sstart'}
writetable(T,'Validation_x1_1500_x2_80.csv')
