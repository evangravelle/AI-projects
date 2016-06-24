% Prob % Prob 5.3k
clear; clc; close all

num_iter = 20;
x = zeros(1,num_iter);
x(1) = 0;
for i = 1:num_iter
    term = 0;
    for k = 1:10
        term = term + 0.05*(exp(x(i)+1/sqrt(k^2+1))-exp(-x(i)-1/sqrt(k^2+1)))/...
          cosh(x(i)+1/sqrt(k^2+1));
    end
    x(i+1) = x(i) - term;
end

hold on
plot(x)
disp(x)
title('5.3k')