function [Hx] = fht(x, ordering)
% FHT  Our implementation of the fast Hadamard transform
%
% We use the approach outlined in "Fast Dimension Reduction..." by Ailon + Liberty
% There are additionaly details in "Unified Matrix Treatment of the Fast Walsh-Hadamard Transform" by Fino + ...
%
% Inputs: x        Input vector (or matrix) of size [2^m k] for some m
%         ordering 'hadamard'/'natural' or 'sequency' (<- not implemented)
%
% Outputs: Hx    Hadamard transform of x using the order prescribed by ordering
%
% Deps: signal/bitrevorder?

%TODO handle other orderings (e.g. sequency)

if nargin < 2
   ordering = 'hadamard';
end

[m,n] = size(x);
[bool, e] = ispow2(m);
if ~bool
   error('fht: Invalid transform length.  Transform dimensions should be powers of 2.');
end

Hx = fht_rec(x, e, ordering);
Hx = Hx / sqrt(2)^e;

end

function [Hx] = fht_rec(x, e, ordering)
% FWT_REC  Recursive implementation of FHT
%
% Note that this doesn't multiply by any 1/sqrt(2) factors

   switch ordering
   case {'hadamard', 'natural'}
      if e == 0
         error('fht_rec: Invalid transform length.  Transform dimensions should be powers of 2.');
      
      elseif e == 1
         Hx = [1 1; 1 -1]*x;
      
      elseif e ==2
         Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;
      
      else
         Hx = zeros(size(x));
         half = pow2(e-1);
         x1 = x(1:half,:); % don't know if recursion will pass by reference or copy... prolly by ref
         x2 = x(half+1:2*half,:);
         Hx1 = fht_rec(x1, e-1, ordering);
         Hx2 = fht_rec(x2, e-1, ordering);
         Hx(1:half,:) = Hx1 + Hx2;
         Hx(half+1:2*half,:) = Hx1 - Hx2;
      end
   
   case {'sequency'}
      %TODO? see eqution (11) of Fino + Algazi (1976).
      error('fht_rec: not implemented');
   
   otherwise
      error('fht_rec: unsupported ordering: %s', ordering);
   end

end
