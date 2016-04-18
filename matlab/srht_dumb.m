% Stolen from http://arxiv.org/pdf/1505.07570v6.pdf
% This is NOT an efficient implementation
% Computes C = A*S, where S is the SRHT matrix
function [C] = srht(A, s)
n = size(A,2);
sgn = randi(2, [1,n])*2 - 3;% one half are +1 and the rest are -1
A = bsxfun(@times, A, sgn); % flip the signs of each colun w.p. 50%
n = 2^(ceil(log2(n)));
C = (fwht(A', n))'; % fast Walsh-Hadamard transform (matlab built-in)
idx = sort(randsample(n,s));
C = C(:, idx); % subsampling
C = C*(n/sqrt(s));
end
