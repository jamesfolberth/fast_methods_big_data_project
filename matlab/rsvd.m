function [U,D,V] = rsvd(A, l, k, G, q)
% RSVD  randomized singular value decomposition
% This function uses the basic randomized algorithm to compute an approximate
% low-rank SVD of a given matrix A.  In other words, it computes an
% approximate factorization
%
%     A \approx U * D * V'
%
% where U and V are orthonormal matrices with k columns, and D is diagonal.
%
% Inputs:  A   Given matrix to be factorized
%          l   Target rank k + oversampling p
%          k   Target rank k
%          G   Sampling matrix or funciton handle that applies the sampling
%              matrix to A: A*G
%          q   Number of power iterations (optional, default: 0)
%
% Outputs: U   Factors in approximate SVD
%          D   ...
%          V   ...
%
% Note: In this code, the caller must supply the sampling matrix (or function).
%       This is done artificially to make testing across several k and
%       different sampling matrices consistent.

% Handle optional input args
if nargin < 5
   q = 0;
end

if isa(G, 'function_handle')
   Y = G(A,l); % computes A*G with l=k+p samples
else
   Y = A*G(:,1:l);
end

[Q,~,~] = qr(Y, 0);          % economy QR

for i = 1:q                  % power iterations
   [Q,~,~] = qr(A'*Q, 0);
   [Q,~,~] = qr(A*Q, 0);
end

B = Q'*A;
[Uhat,D,V] = svd(B, 'econ'); % economy SVD
U = Q*Uhat(:,1:k);
D = D(1:k,1:k);
V = V(:,1:k);

end
