function [Hx] = myfwht(x)
% MYFWHT  Our implementation of the fast Walsh-Hadamard transform
%
% We use the approach outlined in "Fast Dimension Reduction..." by Ailon + Liberty
% There are additionaly details in "Unified Matrix Treatment of the Fast Walsh-Hadamard Transform" by Fino + ...
%
% Deps: signal/bitrevorder?

%TODO handle other orderings (e.g. sequency)

[m,n] = size(x);
[bool, e] = ispow2(m);
if ~bool
   error('Invalid transform length.  Transform dimensions should be powers of 2.');
end

Hx = myfwht_rec(x, e);
Hx = Hx / sqrt(2)^e;

end

function [Hx] = myfwht_rec(x, e)
% MYFHWT_REC  Recursive implementation of MYFWHT (natural/Hadamard order)
%
% Note that this doesn't multiply by any 1/sqrt(2) factors

if e == 0
   error('you dun goofed.');

elseif e == 1
   Hx = [1 1; 1 -1]*x;

elseif e ==2
   Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;

else
   Hx = zeros(size(x));
   half = pow2(e-1);
   x1 = x(1:half,:); % don't know if recursion will pass by reference or copy... prolly by ref
   x2 = x(half+1:2*half,:);
   Hx1 = myfwht_rec(x1, e-1);
   Hx2 = myfwht_rec(x2, e-1);
   Hx(1:half,:) = Hx1 + Hx2;
   Hx(half+1:2*half,:) = Hx1 - Hx2;
end

end


%% Local utility functions
function [bool,e] = ispow2(n)
% ISPOW2  Return true if n is a power of 2, false otherwise.
%
% Inputs: n      integer
%
% Outputs: bool  Boolean true if n == 2^e for some integer e
%          e     integer (if n is not dyadic, e = inf)

   bool = false; e = inf;
   [f,e] = log2(n);
   if f == 0.5
      bool = true;
      e = e-1;
   end
end


function [Z] = fast_kron(A, B, X)
% FAST_KRON  Fast matrix-matrix product kron(A,B)*X
%
% Compute the product kron(A,B)*X for square A and square B.
% 
% Inputs: A    matrix of size [n n]
%         B    matrix of size [m m]
%         X    matrix of size [m*n k]
%
% Outputs: Z   matrix of size [m*n k] equal to kron(A,B)*X but computed
%              more efficiently.
%
% Algorithm:   The algorithm used here is described in 
%              "Unified Matrix Treatment of the Fast Walsh-Hadamard Transform", Fino, 1976.

[mA,nA] = size(A);
[mB,nB] = size(B);
[mX,nX] = size(X);

if mA ~= nA || mB ~= nB
   error('fast_kron: fast_kron only works for square A, B.');
end

if mX ~= nA*nB
   error('fast_kron: size of kron(A,B) and size(X,1) don''t match up.');
end

% This is the algorithm described in "Unified Matrix Treatment of ...", Fino, 1976.
n = nA; % == mA
m = mB; % == nB
Y = zeros(mX,nX);
Z = zeros(mX,nX);
inds1 = 1:n:m*n; % these handle the permutations
inds2 = 1:m;
for i=1:n
   Y(inds1+i-1,:) = B*X(inds2+m*(i-1),:);
end

inds1 = 1:m:n*m; % these handle the permutations
inds2 = 1:n;
for i=1:m
   Z(inds1+i-1,:) = A*Y(inds2+n*(i-1),:);
end

end

function [] = fast_kron_check()
% FAST_KRON_CHECK  A simple test function for dev/verification of fast_kron

m = randi([1 100])
n = randi([1 100])
A = randn(n,n);
B = randn(m,m);

X = randn(m*n,randi([1 100]));
tic
Z_check = kron(A,B)*X;
toc
tic
Z = fast_kron(A,B,X);
toc
norm(Z_check-Z,'inf')

end
