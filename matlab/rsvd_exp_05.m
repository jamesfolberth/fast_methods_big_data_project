function [] = rsvd_exp_05()
% RSVD_EXP_05  RSVD experiment 05 driver
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
q = 6;
t = ceil(log2(n)./q); % t just sufficiently large so n < 2^r == 2^(t*q)
[A,ss] = devils_stairs(n);

%%S = load('mats/Kohonen.mat');
%S = load('mats/pcb3000.mat');
%A = S.Problem.A; [m,n] = size(A);
%qvec = 5:8; % l must be suff. large: l > r >= ceil(log2(n))
%tvec = ceil(log2(n)./qvec); % t just sufficiently large so n < 2^r == 2^(t*q)
%p = 5;

power_its = 0;

% Run accuracy tests for different values of l-p with p fixed
% The SCM matrices change here, so the Gaussian matrices do too.
l = 2^q-1;
k = l-p;
l = 2^q-1;
rad = -2*randi([0 1],n,1) + 1;
scm_sub = randperm(2^(t*q))-1; % message length 2^(t*q)
 
G = randn(n,l); 
[U,D_gauss,V] = rsvd(A,k+p,k,G,power_its);

% Build dual BCH SCM sampler
G_dbch = @(A,l) dbch_sampler(A,l,q,t,rad,scm_sub(1:n));

% dual BCH with 0 power its
[U,D_dbch,V] = rsvd(A,k+p,k,G_dbch,power_its);

% SRFT
dd = exp(1i*2*pi*rand(1,n));
[~,ind] = sort(rand(1,n));
G_srft = @(A,l) srft_sampler(A,l,dd,ind);
[U,D_srft,V] = rsvd(A,l,k,G_srft,power_its);

% SRHT
rad = -2*randi([0 1],n,1) + 1;
[~,ind] = sort(rand(1,n));
G_srht = @(A,l) srht_sampler(A,l,rad,ind);
[U,D_srht,V] = rsvd(A,l,k,G_srht,power_its);

% Compute the errors from truncating the SVD.
%ss  = svd(A)
%ss  = svds(A,l+1);
%ssf = sqrt(triu(ones(length(ss)))*(ss.*ss));

% Create sval plot
figure(1)
clf();
hold on
semilogy(1:l, ss(1:l), 'k-');
semilogy(1:k,diag(D_gauss),'r-o');
semilogy(1:k,diag(D_dbch),'b-+');
semilogy(1:k,diag(D_srft),'m-^');
semilogy(1:k,diag(D_srht),'g-*');
hold off;
set(gca, 'yscale', 'log');
xlabel('k');
ylabel('Singular values');
title(sprintf('Singular values (q=%d)', power_its));
legend('svd','Gaussian','dual BCH','SRFT','SRHT');

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


function [A,s] = devils_stairs(n)
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
