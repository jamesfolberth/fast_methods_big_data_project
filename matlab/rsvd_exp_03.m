function [] = rsvd_exp_03()
% RSVD_EXP_02  RSVD experiment 02 driver
%
% Using Gaussian and dual BCH SCM matrices with fixed oversampling
% We don't get to choose our target ranks as freely now (k = l-p = 2^q-1-p)
% Here we pick a few l values.
% Take a look at Fig 1 of Ubaru et al. TODO: how did they get l to be not of the form 2^q-1?

rng('default'); rng(0); % set generator and seed

% Problem parameters
m = 100;
n = 140;
p = 5;

% Construct a test matrix
%A = LOCAL_fast_decay(m,n,200);
A = LOCAL_slow_decay(m,n);


% Run accuracy tests for different values of l-p with p fixed
% The SCM matrices change here, so the Gaussian matrices do too.
qvec = 4:6;
kvec = 2.^qvec-1-p;
err_gauss = zeros(1,length(qvec));
err_gauss_f = zeros(1,length(qvec));
err_gauss_1 = zeros(1,length(qvec));
err_gauss_1_f = zeros(1,length(qvec));
err_gauss_2 = zeros(1,length(qvec));
err_gauss_2_f = zeros(1,length(qvec));

err_dbch = zeros(1,length(qvec));
err_dbch_f = zeros(1,length(qvec));
err_dbch_1 = zeros(1,length(qvec));
err_dbch_1_f = zeros(1,length(qvec));
err_dbch_2 = zeros(1,length(qvec));
err_dbch_2_f = zeros(1,length(qvec));

for j = 1:numel(qvec)
   q = qvec(j);
   l = 2^q-1;
   t = ceil(log2(n)/q); % t just sufficiently large so n < 2^r == 2^(t*q)
   rad = -2*randi([0 1],n,1) + 1;
   scm_sub = randperm(2^(t*q))-1; % message length 2^(t*q)
   
   k = l-p; % fixed oversampling parameter p
   
   % Build Gaussian sampler
   G = randn(n,l); 

	% Gaussian with 0 power its
   [U,D,V] = rsvd(A,k+p,k,G);
  	err_gauss(j)  = norm(A - U*D*V');
  	err_gauss_f(j) = norm(A - U*D*V','fro');
	
	% Gaussian with 1 power its
  	[U,D,V]  = rsvd(A,k+p,k,G,1);
  	err_gauss_1(j)  = norm(A - U*D*V');
  	err_gauss_1_f(j) = norm(A - U*D*V','fro');

	% Gaussian with 2 power its
  	[U,D,V]  = rsvd(A,k+p,k,G,2);
  	err_gauss_2(j)  = norm(A - U*D*V');
  	err_gauss_2_f(j) = norm(A - U*D*V','fro');

   % Build dual BCH SCM sampler
   G_dbch = @(A,l) dbch_sampler(A,l,q,t,rad,scm_sub(1:n));

	% dual BCH with 0 power its
   [U,D,V] = rsvd(A,k+p,k,G_dbch);
  	err_dbch(j)  = norm(A - U*D*V');
  	err_dbch_f(j) = norm(A - U*D*V','fro');
	
	%% Gaussian with 1 power its
  	%[U,D,V]  = rsvd(A,k,p,G,1);
  	%err_gauss_1(j)  = norm(A - U*D*V');
  	%err_gauss_1_f(j) = norm(A - U*D*V','fro');

	%% Gaussian with 2 power its
  	%[U,D,V]  = rsvd(A,k,p,G,2);
  	%err_gauss_2(j)  = norm(A - U*D*V');
  	%err_gauss_2_f(j) = norm(A - U*D*V','fro');


end

% Compute the errors from truncating the SVD.
ss  = svd(A);
ssf = sqrt(triu(ones(length(ss)))*(ss.*ss));

% Create error plots.
figure(1)
subplot(1,2,1)
hold off
semilogy(0:(length(ss)-1),ss,'k-',...
         kvec,err_gauss,'r-o',...
         kvec,err_gauss_1,'b-o',...
         kvec,err_gauss_2,'m-o',...
         kvec,err_dbch,'g-o',...
         'LineWidth',2)
axis([0,kvec(end),ss(kvec(end)+1),ssf(1)])
xlabel('k')
ylabel('||A - A_k||')
legend('svd','rsvd','rsvdpower q=1','rsvdpower q=2')
title('Spectral norm errors')
subplot(1,2,2)
hold off
semilogy(0:(length(ss)-1),ssf,'k-',...
         kvec,err_gauss_1_f,'r-o',...
         kvec,err_gauss_2_f,'b-o',...
         kvec,err_gauss_2_f,'m-o',...
         kvec,err_dbch_f,'g-o',...
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

