function [bool,e] = ispow2(n)
% ISPOW2  Return true if n is a power of 2, false otherwise.
%
% Inputs: n      integer
%
% Outputs: bool  Boolean true if n == 2^e for some integer e
%          e     integer (if n is not dyadic, e = inf)

bool = false; e = inf;
[f,e] = log2(n);
if f == 0.5
   bool = true;
   e = e-1;
end

end
