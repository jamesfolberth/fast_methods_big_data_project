function [] = rsvd_exp_04()
% RSVD_EXP_04  RSVD experiment 04 driver
%
% Test Gaussian, dual BCH, SRFT, and SRHT over varying l.  We use fixed oversampling p.
% Take a look at Fig 1 of Ubaru et al. TODO: how did they get l to be not of the form 2^q-1?
%                                            perhaps by varying p or k?  Not sure; wasn't explained

rng('default'); rng(0); % set generator and seed

%% Problem parameters
%m = 100;
%n = 140;
%p = 5;
%qvec = 4:6;
%tvec = ceil(log2(n)./qvec); % t just sufficiently large so n < 2^r == 2^(t*q)
%
%% Construct a test matrix
%%A = LOCAL_fast_decay(m,n,200);
%A = LOCAL_slow_decay(m,n);

n = 512;
p = 5;
qvec = 5:8;
tvec = ceil(log2(n)./qvec); % t just sufficiently large so n < 2^r == 2^(t*q)
A = devils_stairs(n);

%%S = load('mats/Kohonen.mat');
%S = load('mats/pcb3000.mat');
%A = S.Problem.A; [m,n] = size(A);
%qvec = 5:8; % l must be suff. large: l > r >= ceil(log2(n))
%tvec = ceil(log2(n)./qvec); % t just sufficiently large so n < 2^r == 2^(t*q)
%p = 5;

power_its = 0;

% Run accuracy tests for different values of l-p with p fixed
% The SCM matrices change here, so the Gaussian matrices do too.
kvec = 2.^qvec-1-p;
err_gauss = zeros(1,length(qvec));
err_gauss_f = zeros(1,length(qvec));
%err_gauss_1 = zeros(1,length(qvec));
%err_gauss_1_f = zeros(1,length(qvec));
%err_gauss_2 = zeros(1,length(qvec));
%err_gauss_2_f = zeros(1,length(qvec));

err_dbch = zeros(1,length(qvec));
err_dbch_f = zeros(1,length(qvec));
%err_dbch_1 = zeros(1,length(qvec));
%err_dbch_1_f = zeros(1,length(qvec));
%err_dbch_2 = zeros(1,length(qvec));
%err_dbch_2_f = zeros(1,length(qvec));

err_srft = zeros(1,length(qvec));
err_srft_f = zeros(1,length(qvec));

err_srht = zeros(1,length(qvec));
err_srht_f = zeros(1,length(qvec));


for j = 1:numel(qvec)
   fprintf(1, '\rj = %d of %d            ', j, numel(qvec));
   q = qvec(j);
   l = 2^q-1;
   %t = ceil(log2(n)/q); % t just sufficiently large so n < 2^r == 2^(t*q)
   t = tvec(j);
   rad = -2*randi([0 1],n,1) + 1;
   scm_sub = randperm(2^(t*q))-1; % message length 2^(t*q)
   
   k = l-p; % fixed oversampling parameter p
   
   % Build Gaussian sampler
   G = randn(n,l); 

	% Gaussian with 0 power its
   [U,D,V] = rsvd(A,k+p,k,G,power_its);
  	err_gauss(j)  = normest(A - U*D*V');
  	err_gauss_f(j) = norm(A - U*D*V','fro');
	
	%% Gaussian with 1 power its
  	%[U,D,V]  = rsvd(A,k+p,k,G,1);
  	%err_gauss_1(j)  = normest(A - U*D*V');
  	%err_gauss_1_f(j) = norm(A - U*D*V','fro');

	%% Gaussian with 2 power its
  	%[U,D,V]  = rsvd(A,k+p,k,G,2);
  	%err_gauss_2(j)  = normest(A - U*D*V');
  	%err_gauss_2_f(j) = norm(A - U*D*V','fro');

   % Build dual BCH SCM sampler
   G_dbch = @(A,l) dbch_sampler(A,l,q,t,rad,scm_sub(1:n));

	% dual BCH with 0 power its
   [U,D,V] = rsvd(A,k+p,k,G_dbch,power_its);
  	err_dbch(j)  = normest(A - U*D*V');
  	err_dbch_f(j) = norm(A - U*D*V','fro');
   
   % SRFT
   dd = exp(1i*2*pi*rand(1,n));
   [~,ind] = sort(rand(1,n));
   G_srft = @(A,l) srft_sampler(A,l,dd,ind);
   [U,D,V] = rsvd(A,l,k,G_srft,power_its);
   err_srft(j) = normest(A-U*D*V');
   err_srft_f(j) = norm(A-U*D*V','fro');
   
   % SRHT
   rad = -2*randi([0 1],n,1) + 1;
   [~,ind] = sort(rand(1,n));
   G_srht = @(A,l) srht_sampler(A,l,rad,ind);
   [U,D,V] = rsvd(A,l,k,G_srht,power_its);
   err_srht(j) = normest(A-U*D*V');
   err_srht_f(j) = norm(A-U*D*V','fro');


end
fprintf(1,'\n');

% Compute the errors from truncating the SVD.
%ss  = svd(A)
ss  = svds(A,2.^(qvec(end)+1));
ssf = sqrt(triu(ones(length(ss)))*(ss.*ss));

% Create error plots.
figure(1)
subplot(1,2,1)
hold off
semilogy(0:(length(ss)-1),ss,'k-',...
         kvec,err_gauss,'r-o',...
         kvec,err_dbch,'b-+',...
         kvec,err_srft,'m-^',...
         kvec,err_srht,'g-*',...
         'LineWidth',2)
axis([0,kvec(end),ss(kvec(end)+1),ss(1)])
xlabel('k')
ylabel('||A - A_k||')
legend('svd','Gaussian','dual BCH','SRFT','SRHT')
title(sprintf('Spectral norm errors (q=%d)', power_its))
subplot(1,2,2)
hold off
semilogy(0:(length(ss)-1),ssf,'k-',...
         kvec,err_gauss_f,'r-o',...
         kvec,err_dbch_f,'b-+',...
         kvec,err_srft_f,'m-^',...
         kvec,err_srht_f,'g-*',...
         'LineWidth',2)
xlabel('k')
ylabel('||A - A_k||')
axis([0,kvec(end),ssf(kvec(end)+1),ssf(1)])
legend('svd','Gaussian','dual BCH','SRFT','SRHT')
title(sprintf('Frobenius norm errors (q=%d)', power_its))

end



function [Y] = srft_sampler(A,l,dd,ind)

   Y = A*diag(dd);
   Y = fft(Y,[],2);
   Y = Y(:,ind(1:l));

end

function [Y] = srht_sampler(A,l,dd,ind)
   
   [m,n] = size(A);
   Y = A*diag(dd);
   N = 2^ceil(log2(n)); % zero-pad to next power of 2
   if N > n
      A_pad = A;
      A_pad(:,n+1:N) = 0;
      Y = srht(A_pad.', ind(1:l), 'hadamard').';
   else
      Y = srht(A.', ind(1:l), 'hadamard').';
   end

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


function A = devils_stairs(n)
length = 20;
s = zeros(n,1);
Nst = floor(n/length);
for i=1:Nst
   s(1+length*(i-1):length*i) = -0.6*(i-1);
end
s(length*Nst:end) = -0.6*(Nst-1);
s = 10.^s;
A = orth(randn(n))*diag(s)*orth(randn(n));

end
