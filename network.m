% This function uses backpropagation to train a Nueral Network
function network

% Training Data
 xt = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7; 0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
 yt = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];
 % scatter(x1,x2, 10, 'filled')

%Initialize the weights and biases
rng(5000);
W2 = .5*randn(2,2); 
W3 = .5*randn(3,2);
W4 = .5*randn(2,3);
b2 = .5*randn(2,1);
b3 = .5*randn(3,1);
b4 = .5*randn(2,1);

eta = .05; %Set learning rate
MaxIter = 1e6; %Set Max Interations
savecost = zeros(MaxIter, 1); %initializes the vector of cost values

%Doing Backpropagation
for i = 1:MaxIter
    k = randi(10);
    x = xt(:, k);
    y=yt(:,k);
    %Forward
    a2 = activate(x, W2, b2);
    a3 = activate(a2, W3, b3);
    a4 = activate(a3, W4, b4);
    %Backwards
    delta4 = a4.*(1-a4).*(a4-y);
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    %Descent Step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4- eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3- eta*delta3;
    b4 = b4- eta*delta4;
    %Monitor Cost
    newcost = cost(W2, W3, W4, b2,b3, b4);%calculate current cost
    savecost(i) = newcost;
    %Display cost every 1000 iterations
    if mod(i,1000) == 1
        i
        newcost
    end
end

save costvec %Save values in costvec file
semilogy([1:1e4:MaxIter],savecost(1:1e4:MaxIter)) %Graphs decay in Cost

%Cost Function
    function C = cost(W2,W3, W4, b2, b3, b4)
        C = 0;
        for j = 1:10
            a2 = activate(xt(:,j), W2, b2);
            a3 = activate(a2, W3, b3);
            a4 = activate(a3, W4, b4);
            C = C + norm(yt(:,j) - a4)^2;
        end
    end
end