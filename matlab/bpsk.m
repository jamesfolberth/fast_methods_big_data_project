function [B] = bpsk(M)
% BPSK  Map {0,1} to {-1, 1}.
% Binary phase-shift keying without the 1/sqrt(2) factor.
% Note that the caller should supply the 1/sqrt(2) factor if needed.
%
% Inputs:  M  message matrix of logical {0,1} entries.
% 
% Outputs: B  message matrix M keyed with BPSK (int8)

B = int8(M==1) - int8(M==0);

end
