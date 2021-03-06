function [] = rsvd_exp_01()
% RSVD_EXP_01  RSVD experiment 01 driver
%
% This driver uses Gaussian sampling matrices and q=0,1,2 power iterations
% This is basically what one of our homeworks was.

rng('default'); rng(0); % set generator and seed

% Problem parameters
m = 100;
n = 140;
p = 5;

% Construct a test matrix
A = LOCAL_fast_decay(m,n,200);
A = LOCAL_slow_decay(m,n);


% Run accuracy tests for different values of k.
% Observe that we use the same matrix of each class (e.g. Gaussian, 
% subsampled SCM) in every experiment.
kvec = 5:5:80;
err_gauss = zeros(1,length(kvec));
err_gauss_f = zeros(1,length(kvec));
err_gauss_1 = zeros(1,length(kvec));
err_gauss_1_f = zeros(1,length(kvec));
err_gauss_2 = zeros(1,length(kvec));
err_gauss_2_f = zeros(1,length(kvec));

% Build Gaussian sampler
G = randn(n,2*n);

for j=1:length(kvec);
   k = kvec(j);
   
	% Gaussian with 0 power its
   [U,D,V] = rsvd(A,k,p,G);
  	err_gauss(j)  = norm(A - U*D*V');
  	err_gauss_f(j) = norm(A - U*D*V','fro');
	
	% Gaussian with 1 power its
  	[U,D,V]  = rsvd(A,k,p,G,1);
  	err_gauss_1(j)  = norm(A - U*D*V');
  	err_gauss_1_f(j) = norm(A - U*D*V','fro');

	% Gaussian with 2 power its
  	[U,D,V]  = rsvd(A,k,p,G,2);
  	err_gauss_2(j)  = norm(A - U*D*V');
  	err_gauss_2_f(j) = norm(A - U*D*V','fro');

end

% Compute the errors from truncating the SVD.
ss  = svd(A);
ssf = sqrt(triu(ones(length(ss)))*(ss.*ss));

% Create error plots.
figure(1)
subplot(1,2,1)
hold off
semilogy(0:(length(ss)-1),ss,'k-',...
         kvec,err_gauss,'r.-',...
         kvec,err_gauss_1,'b.-',...
         kvec,err_gauss_2,'m.-',...
         'LineWidth',2)
axis([0,kvec(end),ss(kvec(end)+1),ssf(1)])
xlabel('k')
ylabel('||A - A_k||')
legend('svd','rsvd','rsvdpower q=1','rsvdpower q=2')
title('Spectral norm errors')
subplot(1,2,2)
hold off
semilogy(0:(length(ss)-1),ssf,'k-',...
         kvec,err_gauss_1_f,'r.-',...
         kvec,err_gauss_2_f,'b.-',...
         kvec,err_gauss_2_f,'m.-',...
         'LineWidth',2)
xlabel('k')
ylabel('||A - A_k||')
axis([0,kvec(end),ss(kvec(end)+1),ssf(1)])
legend('svd','rsvd','rsvdpower q=1','rsvdpower q=2')
title('Frobenius norm errors')

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

