function [] = srht_test()

%rng(0);

%m = 3
%x = (1:2^m).';
%P = (2^m:-1:1).';

m = randi([1 16])
x = randn(2^m, randi([1 100]));
k = randi([1 max(1,2^(m-2))])
p = randperm(2^m); p = p(1:k);
size(x)

tic
%Hx = fwht(x, 2^m, 'hadamard')*sqrt(2)^m; % natural ordering, normalize with 1/2, not 1/sqrt(2)
Hx = fht(x, 'hadamard');
PHx = Hx(p,:);
toc

tic
myPHx = srht(x,p);
toc

%PHx
%myPHx

norm(PHx - myPHx, 'inf')

end
