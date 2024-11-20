%This function calculates the activation at a given layer of a Neural Network.
%The activation function is the sigmoid y = 1/(1+e^x)
%It is applied component wise to the vector Wx + b

function a = activate(x, W, b)
    a = 1./(1+exp(-(W*x+b)));
end