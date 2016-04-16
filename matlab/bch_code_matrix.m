function [Phi] = bch_code_matrix(l,r,inds)
% BCH_CODE_MATRIX  constructs a (subsampled) BCH code matrix
% This function explicitly constructs a full BCH code matrix over {0,1}^l.
% The length must be chosen to form a valid BCH pair.
%
% l should be 2^m-1 for some m <= 16.
% Possible values of r can be found with T = bchnumerr(l), which lists
% values of [l r t], where l is the code length, r is the message length, 
% and t is the number of correctable errors.
% There is no simple formula that enumerates l-r (see e.g. Lin + Costello).
%
% Inputs:  l     code length (should be 2^m-1 for some m <= 16)
%          r     message length (must make a narrow-sense (l,r) BCH code)
%          inds  vector of indexes {0,1,...,2^r-1} (optional, for subsampling)
%
% Outputs: Phi   [2^r l] code matrix or
%                [numel(inds) l] subsampled code matrix
%                Entries are logical 0 or 1.
%
% Deps:    bchenc, bchnumerr from MATLAB Comms System Toolbox

if nargin < 3
   inds = 0:2^r-1;
else
   if ~( all(inds >= 0) && all(inds <= 2^r-1) )
      error('Entries of inds must be in {0,1,...,2^r-1}.');
   end
end

M = logical(uint8(dec2bin(inds,r))-uint8('0')); % message matrix
msg = gf(M, 1);

Phi = bchenc(msg, l, r, 'end');
Phi = logical(Phi.x);

end
