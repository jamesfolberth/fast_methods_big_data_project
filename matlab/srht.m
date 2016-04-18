function [PHx] = srht(x, p, ordering)
% SRHT  Our implementation of the fast subsampled/trimmed Walsh-Hadamard transform
%
% We use the approach outlined in "Fast Dimension Reduction..." by Ailon + Liberty
% There are additionaly details in "Unified Matrix Treatment of the Fast Walsh-Hadamard Transform" by Fino + ...
%
% Inputs: x        Input vector (or matrix) of size [2^m k] for some m
%         p        Subsampling vector of size [2^m 1]
%         ordering 'hadamard'/'natural' or 'sequency' (<- not implemented)
%
% Outputs: PHx    subsampled/trimmed Hadamard transform of x using the order
%                 prescribed by ordering
%
% Note: This smart implementation of the SRHT isn't always faster than the naive
%       implementation of the SRHT using our fht.m:
%           Hx = fht(x, 'hadamard');
%           PHx = Hx(p,:);
%
% Alg: "Fast Dimension Reduction ..." Ailon + Liberty, 2010
%
% Deps: signal/bitrevorder?

%TODO handle other orderings (e.g. sequency)

if nargin < 3
   ordering = 'hadamard';
end

[m,n] = size(x);
[bool, e] = ispow2(m);
if ~bool && e >= 1
   error('srht: Invalid transform length.  Transform dimensions should be 2^e, e >= 1.');
end

[inds, I] = sort(p(:), 1, 'ascend'); % sort p here for faster searching later
Iinv(I) = 1:numel(I); % inverse permutation

PHx = srht_rec(x, e, inds, ordering);
PHx = PHx(Iinv,:) / sqrt(2)^e; % normalization and revert to user's subsampling order

end

function [PHx] = srht_rec(x, e, p, ordering)
% SRHT_REC  Recursive implementation of SRHT
%
% Note that this doesn't multiply by any 1/sqrt(2) factors
%
% Alg: "Fast Dimension Reduction ..." Ailon + Liberty, 2010

   switch ordering
   case {'hadamard', 'natural'}
      if e == 0
         error('srht_rec: Invalid transform length.  Transform dimensions should be 2^e, e >= 1.');
      
      % this case is small enough to compute Hx and then subsample
      elseif e == 1
         Hx = [1 1; 1 -1]*x;
         PHx = Hx(p,:);
      
      % this case is small enough to compute Hx and then subsample
      elseif e ==2
         Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;
         PHx = Hx(p,:);
      
      else
         half = pow2(e-1);
         [p1, p2, m] = split(p, half);
         k1 = numel(p1); k2 = numel(p2);

         x1 = x(1:half,:); % don't know if recursion will pass by reference or copy... prolly by ref
         x2 = x(half+1:2*half,:);
         
         % modify p2 so indexing on next level down matches up with length of x1, x2
         p2_mod = p2 - half;

         PHx = zeros(k1+k2,size(x,2));
         if k1 > 0
            PHx(1:k1,:) = srht_rec(x1+x2, e-1, p1, ordering);
         end

         if k2 > 0
            PHx(k1+1:k1+k2,:) = srht_rec(x1-x2, e-1, p2_mod, ordering);
         end
      end

   case {'sequency'}
      %TODO? see eqution (11) of Fino + Algazi (1976).
      error('srht_rec: not implemented');
   
   otherwise
      error('srht_rec: unsupported ordering: %s', ordering);
   end

end

function [p1, p2, m] = split(p, val)
% SPLIT  split a vector (sorted) along first column at value val via binary search
%
% Inputs: p    sorted vector of unique values
%         val  the value where we want to split
%
% Outputs: m   split index
%          p1  p up through val
%          p2  the rest of p
%
% Note: Due to the convenient indexing behavior in MATLAB, we sometimes will
%       index something as p(1:0), which returns an empty [1 0] array.  If
%       you do size(p(1:0),1) of this [1 0] array, you'll see size 1, even 
%       though the array is empty.  You can use numel(p(1:0)) == 0 or use
%       isempty to determine the array is empty.
%
% Alg: https://en.wikipedia.org/wiki/Binary_search_algorithm

L = 1; R = numel(p);

while true
   if L > R
      % binary search failed
      % assuming p is actually sorted, val may not exist in p
      % we'll go ahead and split at m, which should still split p appropriately
      m = R;
      p1 = p(1:R);
      p2 = p(R+1:end);
      return 
   end

   m = floor((L+R)/2);

   if p(m) < val
      L = m+1;

   elseif p(m) > val
      R = m-1;

   else % p(m) == val, split at m and we're done
      p1 = p(1:m);
      p2 = p(m+1:end);
      return
   end
end

end

