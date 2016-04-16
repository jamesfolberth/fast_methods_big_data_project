function [] = rsvd_exp_02()
% RSVD_EXP_02  RSVD experiment 02 driver
%
% Using Gaussian and dual BCH SCM matrices with fixed oversampling
% We don't get to choose our target ranks as freely now (k = l-p = 2^q-1-p)
% This just uses a fixed l.  Simple experiment to see if dual BCH works.

rng('default'); rng(0); % set generator and seed

% Problem parameters
m = 100;
n = 140;
p = 5;

% Construct a test matrix
%A = LOCAL_fast_decay(m,n,200);
A = LOCAL_slow_decay(m,n);

q = 5;
l = 2^q-1;
t = 2; % t just sufficiently large so n < 2^r == 2^(t*q)
rad = -2*randi([0 1],n,1) + 1;
scm_sub = randperm(2^(t*q))-1; % message length 2^(t*q) where t*q >= ceil(log2(n))

k = l-p; % fixed oversampling parameter p
fprintf('l: %d  k: %d  p: %d\n', l, k, p);

% Build Gaussian sampler
G = randn(n,l); 

% Gaussian with 0 power its
[U,D,V] = rsvd(A,k,p,G);
fprintf('gauss_0:   ||.||_2: %e  ||.||_F: %e\n', norm(A - U*D*V'), norm(A - U*D*V','fro'));

%% Gaussian with 1 power its
%[U,D,V]  = rsvd(A,k,p,G,1);
%err_gauss_1(j)  = norm(A - U*D*V');
%err_gauss_1_f(j) = norm(A - U*D*V','fro');
%
%% Gaussian with 2 power its
%[U,D,V]  = rsvd(A,k,p,G,2);
%err_gauss_2(j)  = norm(A - U*D*V');
%err_gauss_2_f(j) = norm(A - U*D*V','fro');

% Build dual BCH SCM sampler
G_dbch = @(A,k,p) dbch_sampler(A,k,p,q,t,rad,scm_sub(1:n));

% dual BCH with 0 power its
[U,D,V] = rsvd(A,k,p,G_dbch);
fprintf('dbch_0:    ||.||_2: %e  ||.||_F: %e\n', norm(A - U*D*V'), norm(A - U*D*V','fro'));

% Compute the errors from truncating the SVD.
ss  = svd(A);
ssf = sqrt(triu(ones(length(ss)))*(ss.*ss));

fprintf('tsvd:      ||.||_2: %e  ||.||_F: %e\n', ss(k), ssf(k));

end


% These are stolen from Gunnar's HW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function build a test matrix A whose singular values decay
% exponentially. To be precise, it builds a matrix A via
%   A = U * D * V'
% where U and V are randomly drawn ON matrices, and D is diagonal. The
% entries of D are taken to be D(i,i) = beta^(i-1) where beta is chosen so
% that D(k,k) = 1e-15, where k is an input parameter.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_fast_decay(m,n,k)

p       = min(m,n);
[U,~,~] = qr(randn(m,p),0);
[V,~,~] = qr(randn(n,p),0);
beta    = (1e-15)^(1/(k-1));
ss      = beta.^(0:(p-1));
A       = U * diag(ss) * V';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function build a test matrix A whose singular values decay
% slowly. To be precise, it builds a matrix A via
%    A = U * D * V'
% where U and V are randomly drawn ON matrices, and D is diagonal. The
% entries of D are taken to be D(i,i) = 1/i.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_slow_decay(m,n)

p       = min(m,n);
[U,~,~] = qr(randn(m,p),0);
[V,~,~] = qr(randn(n,p),0);
ss      = 1./(1:p);
A       = U * diag(ss) * V';

end

