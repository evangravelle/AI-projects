 clear; clc; close all

% Problem 3.3

num_iter = 1000000;
n = 10;
alpha = 0.25;
count = 0;
den = 0;
num = 0;
for i = 1:num_iter
    B = randi(2,10,1) - 1;
    f_B = 0;
    for j = 1:n
        f_B = f_B + 2^(j-1)*B(j);
    end
    P_Z = (1-alpha)/(1+alpha)*alpha^abs(128-f_B);
    if B(8) == 1
        num = num + P_Z;
    end
    den = den + P_Z;
    P(i) = num/den;
end

plot(1:num_iter,P)
xlabel('Number of samples')
ylabel('Probability of B_8=1 given Z=128')