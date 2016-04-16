function [Phi] = dbch_code_matrix(l,t,inds)
% DBCH_CODE_MATRIX  constructs a (subsampled) dual BCH code matrix
% This function explicitly constructs a full dual BCH code matrix over {0,1}^l.
% The parameters must be chosen to form a valid (primal) BCH pair.
%
% l should be 2^m-1 for some m <= 16.
% Possible values of r can be found with T = bchnumerr(l), which lists
% values of [l r t], where l is the code length, r is the message length, 
% and t is the number of correctable errors.
% There is no simple formula that enumerates l-r (see e.g. Lin + Costello).
%
% Inputs:  l     code length (should be 2^m-1 for some m <= 16)
%          t     number of correctable errors for the primal BCH code
%                t should be relatively small, as the "enumeration" used in
%                Ubaru et al. doesn't hold for large t.
%          inds  vector of indexes {0,1,...,2^(t*m)-1} (optional, for subsampling)
%
% Outputs: dPhi   [2^(t*m) l] code matrix or
%                 [numel(inds) l] subsampled code matrix
%                 Entries are logical 0 or 1.
%
% Note: The number of correctable errors for the primal code, t, should be
%       relatively small.  For small t, there is an approximate enumeration:
%       (2^m-1, 2^m-1-t*m) primal BCH with distance >= 2*t+1
%       (2^m-1, t*m) dual BCH with distance >= 2^(m-1) - (t-1)*2^(m/2)
%
% Deps:    bchenc, bchnumerr, gf/deconv from MATLAB Comms System Toolbox

m = log2(l+1);
pr = l-t*m; % primal message length (a.k.a. dimension)
dr = t*m;   % dual message length (a.k.a. dimension)

if t ~= bchnumerr(l,l-dr)
   error('BCH parameters out of small t regime.');
end

if nargin < 3
   inds = 0:2^dr-1;
else
   if ~( all(inds >= 0) && all(inds <= 2^dr-1) )
      error('Entries of inds must be in {0,1,...,2^(t*m)-1}.');
   end
end

M = logical(uint8(dec2bin(inds,dr))-uint8('0')); % message matrix
msg = gf(M, 1);

% Construct the dual generator polynomial
% http://cstheory.stackexchange.com/questions/475/dual-bch-codes-of-design-distance-d
genpoly = bchgenpoly(l, pr);
xlp1 = gf([1 zeros(1,l-1) 1], 1); % == x^l + x^0 == x^l + 1 == x^l - 1 over GF(2)
[dgenpoly,remainder] = deconv(xlp1, genpoly); % poly division

% modified from bchenc.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[nRows, nCols] = size(msg);

% Set up the shift register
a  = fliplr(logical(dgenpoly.x));  % Extract the value from the gf object
st = length(a)-1;

u = logical(msg.x);

reg = false(nRows, st);
for iCol = 1 : nCols  % Loop over bits in the message words

    % Perform an XOR between the input and the shift register.  Recall that
    % there is one codeword per row.
    d = (u(:,iCol) | reg(:,st)) & ~(u(:,iCol) & reg(:,st));

    for idxPoly = st : -1 : 2

        % For one codeword, the line below is equivalent to
        % If d
        %     reg(idxPoly) = reg(idxPoly-1) || a(idxPoly) && ...
        %                    ~(reg(idxPoly-1) && a(idxPoly));
        % else
        %     reg(idxPoly) = reg(idxPoly-1);
        % end
        % It performs an XOR between the register and the generator polynomial.
        reg(:,idxPoly) = ( reg(:,idxPoly-1) | (d & a(:,idxPoly)) ) & ...
            ( ~d | ~(reg(:,idxPoly-1) & a(:,idxPoly)) );
    end
    reg(:,1) = d;
end

% Rearrange parity if necessary
parity = double(fliplr(reg));
Phi = gf([double(u) parity]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Phi = logical(Phi.x);

end
