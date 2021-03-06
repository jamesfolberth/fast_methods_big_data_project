function [] = fht_test()

%rng(0);

%m = randi([1 16])
%x = rand(2^m,randi([1 100]));
%size(x)
m = 3
x = (1:2^m).';
tic
Hx = fwht(x, 2^m, 'hadamard')*sqrt(2)^m; % natural ordering, normalize with 1/2, not 1/sqrt(2)
toc
tic
myHx = fht(x)
toc

%Hx
%myHx

norm(Hx - myHx, 'inf')

end
