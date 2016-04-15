function [Phi] = dbch_code_matrix(l,r,inds)
% DBCH_CODE_MATRIX  constructs a (subsampled) dual BCH code matrix
%TODO complete help docs.
%TODO how do the code/msg lengths work for dual BCH codes?
% %This function explicitly constructs a full BCH code matrix over {0,1}^l.
% %The length  must be chosen to form a valid pair from
% %  http://www.mathworks.com/help/comm/ref/bchenc.html
%
% %Possible values of r can be found with T = bchnumerr(l), which lists
% %values of [l r t], where l is the code length, r is the message length, 
% %and t is the number of correctable errors.
% %There is no simple formula that enumerates l-r (see Lin + Costello).
%
% %Inputs:  l     code length (should be 2^m-1 for some m <= 16)
% %         r     message length (must make a narrow-sense (l,r) BCH code)
% %         inds  vector of indexes {0,1,...,2^r-1} (optional, for subsampling)
%
% %Outputs: Phi   [2^r l] code matrix or
% %               [numel(inds) l] subsampled code matrix
% %               Entries are logical 0 or 1.
%
% Deps:    bchenc, bchnumerr, gfdeconv from MATLAB Comms System Toolbox


if r > 16
   error('MATLAB''s BCH encoder only supports code lengths <= 2^16-1.  Choose r <= 16');
end

if nargin < 3
   inds = 0:2^r-1;
else
   if ~( all(inds >= 0) && all(inds <= 2^r-1) )
      error('Entries of inds must be in {0,1,...,2^r-1}.');
   end
end

M = logical(uint8(dec2bin(inds,r))-uint8('0')); % message matrix
msg = gf(M, 1);

% Construct the dual generator polynomial
% http://cstheory.stackexchange.com/questions/475/dual-bch-codes-of-design-distance-d
genpoly = bchgenpoly(l, r);
xlp1 = gf([1 zeros(1,l-2) 1], 1); % == x^l + 1
[dgenpoly,r] = deconv(xlp1, genpoly); % poly division

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
