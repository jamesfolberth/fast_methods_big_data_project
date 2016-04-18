function [Y] = dbch_sampler(A,l,q,t,rad,inds)
% DBCH_SAMPLER  Compute A*G where G is a dual BCH SCM matrix
%TODO

% This builds the full subsampled dual BCH matrix and applies it.
% This doesn't scale as well as theoretically possible, but it works.
assert(l == 2^q-1);
r = t*q;
SPhi = dbch_code_matrix(l,t,inds); % elements in {-1,1}
Y = (A*diag(rad))*SPhi/sqrt(l);

end
