clear all;
cd \path\;
rng('default')
rep = 2000;
nu = 0.2;
A = [];
dt = 0.01;tim = 10/dt; sdt = sqrt(dt);
x1 = 1.8;
x2 = 0.5;
tic
for j = 1:1:rep
y = [];
y(1) = 0;
for t=1:tim % Euler-Maruyama
y(t+1) = y(t) + (x1 - y(t))*dt + ((nu*y(t)+ 1)*x2*randn(1)*sdt);
end
A(end+1) = y(tim+1);
end
toc
T = array2table(A');
T.Properties.VariableNames(1) = {'y'};
writetable(T,'validation_sde_x1_1point8_x2_point5.csv');
        
