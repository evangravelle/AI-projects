% Problem 4.6
% Evan Gravelle, Spring 2016
clear;clc;close all

load('C:\Users\evan\Desktop\AI\hw4\newTest3.mat')
load('C:\Users\evan\Desktop\AI\hw4\newTest5.mat')
load('C:\Users\evan\Desktop\AI\hw4\newTrain3.mat')
load('C:\Users\evan\Desktop\AI\hw4\newTrain5.mat')
x = [newTrain3;newTrain5];
x_test = [newTest3;newTest5];
sigma = @(x) 1/(1+exp(-x));

% Maximizing log-likelihood of training examples
% Define y=1 is 3, y=0 is 5
num_iter = 10;
w = zeros(64,num_iter);
L = zeros(num_iter,1);
P_y1 = zeros(size(x,1),num_iter);
y = [ones(1,size(newTrain3,1)) zeros(1,size(newTrain5,1))]';
correct = zeros(num_iter,1);
for k = 1:num_iter
    grad = zeros(64,1);
    H = zeros(64,64);
    for t = 1:size(x,1)
        P_y1(t,k) = sigma(w(:,k)'*x(t,:)');
        grad = grad + (y(t) - P_y1(t,k))*x(t,:)';
        H = H - P_y1(t,k)*(1 - P_y1(t,k))*x(t,:)'*x(t,:);
        L(k) = L(k) + y(t)*log(P_y1(t,k)) + (1-y(t))*log(1-P_y1(t,k));
        
        if t <= size(newTrain3,1)
            correct(k) = correct(k) + round(P_y1(t,k));
        else
            correct(k) = correct(k) + (1 - round(P_y1(t,k)));
        end
        
    end
    
    w(:,k+1) = w(:,k) - H\grad;
end

w_final = w(:,end);
w_print = reshape(w_final',8,8);
disp('The weights have converged to:')
disp(' ')
disp(w_print)
disp(' ')

figure(1)
plot(L)
title('Log-likelihood of Train3')
xlabel('Iteration')
ylabel('L')

figure(2)
plot(100*correct/size(x,1))
title('Percent error of detected 3s and 5s')
xlabel('Iteration')
ylabel('Percent error')

% Testing on new data
correct_test = 0;
P_y1_test = zeros(size(x_test,1),1);
for t = 1:size(x_test,1)
    P_y1_test(t) = sigma(w_final'*x_test(t,:)');
    
    if t <= size(newTest3,1)
        correct_test = correct_test + round(P_y1_test(t));
        if t == size(newTest3,1)
            correct_test3 = correct_test;
        end
    else
        correct_test = correct_test + (1 - round(P_y1_test(t)));
    end
end

disp(['Test percent error is ' num2str(100*correct_test/size(x_test,1)) '%'])
disp(['Threes are detected with ' num2str(100*correct_test3/size(newTest3,1)) '% accuracy'])
disp(['Fives are detected with ' num2str(100*(correct_test-correct_test3)/size(newTest5,1)) '% accuracy'])

