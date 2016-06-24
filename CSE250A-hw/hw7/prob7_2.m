% Problem 7.2
clear;clc;close all

load('data.mat')
load('image.mat')
R = rewards;
n = 81;
gamma = 0.9875;
max_num_iter = 1000;
max_num_iter2 = 20;
V = zeros(n,max_num_iter+1);
I = eye(n);

% Construct transition matrices
A = zeros(n,n,4);
for i = 1:size(prob_a1,1)
    A(prob_a1(i,1),prob_a1(i,2),1) = prob_a1(i,3);
end
for i = 1:size(prob_a2,1)
    A(prob_a2(i,1),prob_a2(i,2),2) = prob_a2(i,3);
end
for i = 1:size(prob_a3,1)
    A(prob_a3(i,1),prob_a3(i,2),3) = prob_a3(i,3);
end
for i = 1:size(prob_a4,1)
    A(prob_a4(i,1),prob_a4(i,2),4) = prob_a4(i,3);
end

% Value iteration
best_action = ones(n,1);
for i = 1:max_num_iter+1
    for j = 1:n
        best_term = -Inf;
        best_action(j) = 0;
        for k = 1:4
            term = A(j,:,k)*V(:,i);
            if term > best_term
                best_term = term;
                best_action(j) = k;
            end
        end

        V(j,i+1) = R(j) + gamma*best_term;
    end
    
    % convergence criterion
    if var(V(:,i+1) - V(:,i)) < 0.000001
        last_iteration = i;
        break;
    end
end

for j = 1:n
    if V(j,last_iteration+1) ~= 0
        fprintf('V(%d) = %.4f\n',int64(j),V(j,last_iteration+1))
    end
end

% Draw arrows in image
% drawArrow = @(x,y) quiver(x(1), y(1), x(2)-x(1), y(2)-y(1), 0) 
drawArrow = @(x,y) line([x(1), y(1)], [x(2), y(2)]) ;
hold on
imshow(maze, colormap, 'InitialMagnification', 50)
box_x_length = size(maze,2)/9;
box_y_length = size(maze,1)/9;
arrow_length = 0.75*box_x_length/2;
for i = 1:n
    if V(i,last_iteration+1) ~= 0 && ~ismember(i,[47 49 51 65 67 69 79])
        center = [box_x_length*ceil(i/9)-box_x_length/2 box_y_length*mod(i-1,9)+box_y_length/2];
        switch best_action(i)
            case 1 % west
                arrow(center + [arrow_length 0], center - [arrow_length 0])
            case 2 % north
                arrow(center + [0 arrow_length], center - [0 arrow_length])
            case 3 % east
                arrow(center -[arrow_length 0], center + [arrow_length 0])
            case 4 % south
                arrow(center - [0 arrow_length], center + [0 arrow_length])
            otherwise
                disp('mistake')
        end
    end
end

% Policy iteration
V2 = zeros(n,max_num_iter2+1);
policy = zeros(n,max_num_iter2+1);
policy(:,1) = 4*ones(n,1); % east
for i = 1:max_num_iter2+1
    P = zeros(n);
    for j = 1:n
        P(j,:) = A(j,:,policy(j,i));
    end
    V2(:,i+1) = (I - gamma*P)\R;
    
    for j = 1:n
        best_term = -Inf;
        for k = 1:4
            term = A(j,:,k)*V2(:,i+1);
            if term > best_term
                best_term = term;
                policy(j,i+1) = k;
            end
        end
    end

    if policy(:,i+1) == policy(:,i)
        last_iteration2 = i;
        break;
    end
end

if all(policy(:,last_iteration2) == best_action)
    disp(' ')
    disp('The optimal policy computed with both methods are the same!')
end

disp('With an initial policy of move east, the policy iteration converges in 5 iterations')
disp('With an initial policy of move west, the policy iteration also converges in 5 iterations')
disp('Note, with an initial policy of move south, the policy iteration converges in 11 iterations')