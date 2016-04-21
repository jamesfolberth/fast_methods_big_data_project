function [] = fixed_mat_driver();

%rng(1);

%A = randn(140,100);
%l = 31;
%errs = comparison_01(A,l,l,5);

%m = 100;
%n = 140;
%p = 5;
%%A = LOCAL_fast_decay(m,n,200);
%A = LOCAL_slow_decay(m,n);
%l = 31;
%errs = comparison_01(A,l,l-p,5);

% Kohonen
%A = load_mat(0);
%l = 511;

% EPA
%A = load_mat(1);
%l = 255;

% lpi_ceria3d
%A = load_mat(2);
%l = 63;
%l = 31;

% S80PI_n1
%A = load_mat(3);
%l = 63;

% delaunay_n12
%A = load_mat(4);
%l = 63;

% deter3
A = load_mat(5); % I have issues running this with parfor
l = 127;

n_samples = 25;
errs = zeros(5,n_samples);
for i=1:n_samples
   %errs(:,i) = comparison_01(A,l,l-5,ceil(log2(l)));
   errs(:,i) = comparison_02(A,l,ceil(log2(l)));
end
[m,s] = simple_err_stats(errs);


end

function [errs] = comparison_01(A,l,k,q)
% {{{
% COMPARISON_01  Try to recreate Table 1 of Ubaru et al. 2015 but using the RSVD error
%
% Spectral norm
% 
% errs = ['svds', 'Gaussian', 'dual BCH', 'SRFT', 'SRHT']

[m,n] = size(A);

errs = [];

% svds
s = svds(A,k+1);
errs = [errs; s(k+1)];
fprintf(1,'svds:        %e\n',s(k+1));

% Gaussian
G = randn(n,l);
[U,D,V] = rsvd(A,l,k,G);
errs = [errs; norm(A-U*D*V')];
fprintf(1,'Gaussian:    %e\n',errs(end));

% dual BCH
t = ceil(log2(n)/q); % t just sufficiently large so n < 2^r == 2^(t*q)
rad = -2*randi([0 1],n,1) + 1;
scm_sub = randperm(2^(t*q))-1; % message length 2^(t*q)
G_dbch = @(A,l) dbch_sampler(A,l,q,t,rad,scm_sub(1:n));
[U,D,V] = rsvd(A,l,k,G_dbch);
errs = [errs; norm(A-U*D*V')];
fprintf(1,'dual BCH:    %e\n',errs(end));

% SRFT
dd = exp(1i*2*pi*rand(1,n));
[~,ind] = sort(rand(1,n));
G_srft = @(A,l) srft_sampler(A,l,dd,ind);
[U,D,V] = rsvd(A,l,k,G_srft);
errs = [errs; norm(A-U*D*V')];
fprintf(1,'SRFT:        %e\n',errs(end));

% SRHT
rad = -2*randi([0 1],n,1) + 1;
[~,ind] = sort(rand(1,n));
G_srht = @(A,l) srht_sampler(A,l,rad,ind);
[U,D,V] = rsvd(A,l,k,G_srht);
errs = [errs; norm(A-U*D*V')];
fprintf(1,'SRHT:        %e\n',errs(end));


% }}}
end

function [errs] = comparison_02(A,l,q)
% {{{
% COMPARISON_02  Try to recreate Table 1 of Ubaru et al. 2015 using just the projection error
%
% Spectral norm
% 
% errs = ['svds', 'Gaussian', 'dual BCH', 'SRFT', 'SRHT']

[m,n] = size(A);

errs = [];

% svds
s = svds(A,l+1);
errs = [errs; s(l+1)];
fprintf(1,'svds:        %e\n',s(l+1));

% Gaussian
G = randn(n,l);
errs = [errs; proj_error(A,l,G)];
fprintf(1,'Gaussian:    %e\n',errs(end));

% dual BCH
t = ceil(log2(n)/q); % t just sufficiently large so n < 2^r == 2^(t*q)
rad = -2*randi([0 1],n,1) + 1;
scm_sub = randperm(2^(t*q))-1; % message length 2^(t*q)
G_dbch = @(A,l) dbch_sampler(A,l,q,t,rad,scm_sub(1:n));
errs = [errs; proj_error(A,l,G_dbch)];
fprintf(1,'dual BCH:    %e\n',errs(end));

% SRFT
dd = exp(1i*2*pi*rand(1,n));
[~,ind] = sort(rand(1,n));
G_srft = @(A,l) srft_sampler(A,l,dd,ind);
errs = [errs; proj_error(A,l,G_srft)];
fprintf(1,'SRFT:        %e\n',errs(end));

% SRHT
rad = -2*randi([0 1],n,1) + 1;
[~,ind] = sort(rand(1,n));
G_srht = @(A,l) srht_sampler(A,l,rad,ind);
errs = [errs; proj_error(A,l,G_srht)];
fprintf(1,'SRHT:        %e\n',errs(end));


% }}}
end



function [m,s] = simple_err_stats(errs)
   
   m = mean(errs,2);
   s = std(errs,0,2);
   
   fprintf(1, 'Simple Error Stats:\n');
   fprintf(1,'svds:        %e +- %e\n',m(1),s(1));
   fprintf(1,'Gaussian:    %e +- %e\n',m(2),s(2));
   fprintf(1,'dual BCH:    %e +- %e\n',m(3),s(3));
   fprintf(1,'SRFT:        %e +- %e\n',m(4),s(4));
   fprintf(1,'SRHT:        %e +- %e\n',m(5),s(5));
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

function [A] = load_mat(ind)
% 0 - mats/Kohonen.mat
% 1 - mats/EPA.mat
% 2 - mats/lpi_ceria3d.mat
% 3 - mats/S80PI_n1.mat
% 4 - mats/denaunay_n12.mat
% 5 - mats/deter3.mat
%
% Get help via >> help fixed_mat_driver>load_mat

switch ind
case 0
   fprintf(1,'Kohonen.mat\n');
   S = load('mats/Kohonen.mat');
   A = S.Problem.A;

case 1
   fprintf(1,'EPA.mat\n');
   S = load('mats/EPA.mat');
   A = S.Problem.A;

case 2
   fprintf(1, 'lpi_ceria3d.mat\n');
   S = load('mats/lpi_ceria3d.mat');
   A = S.Problem.A;

case 3
   fprintf(1, 'S80PI_n1.mat\n');
   S = load('mats/S80PI_n1.mat');
   A = S.Problem.A;

case 4
   fprintf(1, 'delaunay_n12.mat\n');
   S = load('mats/delaunay_n12.mat');
   A = S.Problem.A;

case 5
   fprintf(1, 'deter3.mat\n');
   S = load('mats/deter3.mat');
   A = S.Problem.A;

otherwise
   error('bad index.');
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

