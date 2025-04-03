%%% Image deblurring: min ||Ax-b||^2+lam*||x||_1%%%

clc;
clear;

dim = 256; % using 256x256 photo
n=dim^2;
m=n;

RGB=double(imread('cameraman.jpg')); % read image and change to double
original=(RGB(:,:,1))/255; % scaling
fun=@(u) imgaussfilt(u,1); % filter
A = func2mat(fun,original); % change the filter operator to matrix
b=(A)*original(:); % blurred image

x = b; % set strtaing point as the blurred photo
y = x; 
t = 1;
lam = 1e-4; % regularization parameter
alpha = 1/(2*normest(A)^2); % stepsize 1/L

%%% FISTA Algorithm

for k = 1:100
    grad = 2*A'*(A*y-b); % gradinet at point y
    xold = x;
    x = y - alpha*grad; % gradinet step
    x = max(0,abs(x)-alpha*lam).*sign(x); % proximal step
    told = t;
    t = (1+sqrt(1+4*t^2))/2;
    y = x + (told-1)/t*(xold-x); % acceleration
end

x = reshape(x,dim,dim); % change the vector to photo (x is the output of FISTA)
b = reshape(b,dim,dim); 
montage({b,x}); % show image
