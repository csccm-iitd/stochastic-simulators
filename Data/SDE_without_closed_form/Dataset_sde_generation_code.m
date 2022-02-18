clear all;
cd \path\;
rng('default');
num = 60;
rep = 10;
nu = 0.2;
A = [];
dt = 0.01;tim = 10/dt; sdt = sqrt(dt);
for i = 1:1:num
x1 = 0.9 + (2-0.9)*rand(1);
x2 = 0.1 + (1-0.1)*rand(1);
for j = 1:1:rep
y = [];
y(1) = 0;
for t=1:tim % Euler-Maruyama
y(t+1) = y(t) + (x1 - y(t))*dt + ((nu*y(t)+ 1)*x2*randn(1)*sdt);
end
A(end+1,1) = x1;
A(end,2) = x2;
A(end,3) = y(tim+1);
end
end
T = array2table(A);
T.Properties.VariableNames(1:3) = {'x1_train','x2_train','y_train'};
writetable(T,'SDE60X10.csv');
        
        
