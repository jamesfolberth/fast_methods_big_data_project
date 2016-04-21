function [err] = proj_error(A, l, G)
% PROJ_ERROR compute error due to projection in RSVD
%
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

if isa(G, 'function_handle')
   Y = G(A,l); % computes A*G with l=k+p samples
else
   Y = A*G(:,1:l);
end

[Q,~,~] = qr(Y, 0);          % economy QR

err = normest(A-Q*(Q'*A)); % default rel tol of 1e-6

end
