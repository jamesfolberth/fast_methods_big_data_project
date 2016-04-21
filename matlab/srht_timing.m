function [] = srht_timing()
% SRHT_TIMING  Simple timing experiments to verify the scaling of SRHT
%
% Theoretical timing is O(d*log(k)) for length d naturally ordered Hadamard
% transform and picking k elements.

experiment_01();
%experiment_02();
%experiment_03(); % I don't really like this.  experiment 01 is clear enough


end

function [PHx] = srht_naive(x,p)
% SRHT_NAIVE  Simple implementation of SRHT using our FHT
   Hx = fht(x,'hadamard');
   %Hx = fwht(x,size(x,1),'hadamard');
   PHx = Hx(p,:);
end

function [] = experiment_01()
% {{{
m_vec = 11:14;
num_samples = 10;
x_2nd_dim = 20;

% run and collect computation times
k_vecs = cell(numel(m_vec),1);
naive_times = cell(numel(m_vec),1);
srht_times = cell(numel(m_vec),1);
for i=1:numel(m_vec)
   m = m_vec(i);
   d = pow2(m);

   x = randn(d,x_2nd_dim);
   
   k_vecs{i} = floor(2.^(linspace(4,m,15))); % build our own logspace
   naive_times{i} = zeros(num_samples, numel(k_vecs{i}));
   srht_times{i} = zeros(num_samples, numel(k_vecs{i}));

   for j=1:numel(k_vecs{i})
      k = k_vecs{i}(j);
      fprintf(1,'\r m = %d, k = %d                             ', m, k);
      for n=1:num_samples
         p = randperm(d,k);
         %[p,~] = sort(p);

         tic();
         PHx = srht_naive(x,p);
         t = toc();
         naive_times{i}(n,j) = t;

         tic();
         PhX = srht(x,p);
         t = toc();
         srht_times{i}(n,j) = t;
      end
   end
end
fprintf(1,'\n');

% plot things
figure(1);
clf();
hold on;
for i=1:numel(m_vec);
   %m = m_vec(i);
   %avg = mean(naive_times{i},1)
   %stddev = std(naive_times{i},0,1);
   %errorbar(k_vecs{i}, avg, stddev, 'b--');
   
   avg = mean(naive_times{i}(:)); % just plot average time as a constant, per Gunnar's recommendation
   plot([k_vecs{i}(1) k_vecs{i}(end)], [avg avg], 'g--');

   m = m_vec(i);
   avg = mean(srht_times{i},1);
   stddev = std(srht_times{i},0,1);
   errorbar(k_vecs{i}, avg, stddev, 'b-');
end
set(gca, 'xscale', 'log');
xlabel('k - number of subsamples');
ylabel('runtime (s)');
title('Runtime scaling of SRHT vs. k (various d)');
legend('Simple SRHT', 'Better SRHT', 'Location', 'SouthEast');
hold off;

% }}}
end


function [] = experiment_02()
% {{{
k_vec = floor(2.^(linspace(5,12,7)));
max_m = 14;
num_samples = 10;
x_2nd_dim = 20;

% run and collect computation times
m_vecs = cell(numel(k_vec),1);
naive_times = cell(numel(k_vec),1);
srht_times = cell(numel(k_vec),1);
for i=1:numel(k_vec)
   k = k_vec(i);

   m_vecs{i} = ceil(log2(k)):max_m;
   naive_times{i} = zeros(num_samples, numel(m_vecs{i}));
   srht_times{i} = zeros(num_samples, numel(m_vecs{i}));

   for j=1:numel(m_vecs{i})
      m = m_vecs{i}(j);
      d = pow2(m);
      x = randn(d,x_2nd_dim);
      fprintf(1,'\r k = %d, m = %d                             ', k, m);
      for n=1:num_samples
         p = randperm(d,k);
         %[p,~] = sort(p);

         tic();
         PHx = srht_naive(x,p);
         t = toc();
         naive_times{i}(n,j) = t;

         tic();
         PHX = srht(x,p);
         t = toc();
         srht_times{i}(n,j) = t;
      end
   end
end
fprintf(1,'\n');

% plot things
figure(1);
clf();
hold on;
for i=1:numel(k_vec);
   avg = mean(naive_times{i},1);
   stddev = std(naive_times{i},0,1);
   errorbar(pow2(m_vecs{i}), avg, stddev, 'g--');
   
   avg = mean(srht_times{i},1);
   stddev = std(srht_times{i},0,1);
   errorbar(pow2(m_vecs{i}), avg, stddev, 'b-');
end
set(gca, 'xscale', 'log', 'yscale', 'log');
xlabel('d - length of transform');
ylabel('runtime (s)');
title('Runtime scaling of SRHT vs. d (various k)');
legend('Simple SRHT', 'Better SRHT', 'Location', 'SouthEast');
hold off;

% }}}
end


function [] = experiment_03()
% {{{
m_vec = 11:14;
num_samples = 10;
x_2nd_dim = 20;

% run and collect computation times
k_vecs = cell(numel(m_vec),1);
naive_times = cell(numel(m_vec),1);
srht_times = cell(numel(m_vec),1);
for i=1:numel(m_vec)
   m = m_vec(i);
   d = pow2(m);

   x = randn(d,x_2nd_dim);
   
   k_vecs{i} = floor(2.^(linspace(4,m,15))); % build our own logspace
   naive_times{i} = zeros(num_samples, numel(k_vecs{i}));
   srht_times{i} = zeros(num_samples, numel(k_vecs{i}));

   for j=1:numel(k_vecs{i})
      k = k_vecs{i}(j);
      fprintf(1,'\r m = %d, k = %d                             ', m, k);
      for n=1:num_samples
         p = randperm(d,k);
         %[p,~] = sort(p);

         tic();
         PHx = srht_naive(x,p);
         t = toc();
         naive_times{i}(n,j) = t;

         tic();
         PhX = srht(x,p);
         t = toc();
         srht_times{i}(n,j) = t;
      end
   end
end
fprintf(1,'\n');

% plot things
figure(1);
clf();
hold on;
for i=1:numel(m_vec);
   %m = m_vec(i);
   %avg = mean(naive_times{i},1)
   %stddev = std(naive_times{i},0,1);
   %errorbar(k_vecs{i}, avg, stddev, 'b--');
   
   avg = mean(naive_times{i}(:)); % just plot average time as a constant, per Gunnar's recommendation
   plot([k_vecs{i}(1) k_vecs{i}(end)], [avg avg], 'g--');

   m = m_vec(i);
   avg = mean(srht_times{i},1);
   stddev = std(srht_times{i},0,1);
   errorbar(k_vecs{i}, avg./log10(k_vecs{i}), stddev./log10(k_vecs{i}), 'b-');
end
set(gca, 'xscale', 'log');
xlabel('k - number of subsamples');
ylabel('runtime (s)');
title('Runtime scaling of SRHT vs. k (various d)');
legend('Simple SRHT', 'Better SRHT', 'Location', 'SouthEast');
hold off;

% }}}
end



