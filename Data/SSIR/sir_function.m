function s = sir_function(beta,gamma,S,I,R,N)
while I>0
 a = beta*S*I/N; b = gamma*I; p1 = a/(a+b); u1 = rand(1); 
 if u1>0 && u1<=p1
     S = S-1; I= I + 1; R = R;
 elseif u1>p1 && u1 < 1
     S = S;I = I - 1; R = R+1;
 end
end
s = S;  
end