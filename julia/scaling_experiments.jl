
using PyPlot

include("fht.jl")
include("srht.jl")

function experiment_01()
# {{{
m_vec = collect(12:14);
#m_vec = collect(6:8);
num_samples = 10;
x_2nd_dim = 20;

# run and collect computation times
k_vecs = cell(length(m_vec),1);
naive_times = cell(length(m_vec),1);
srht_times = cell(length(m_vec),1);
for i=1:length(m_vec)
   m = m_vec[i];
   d = 2<<(m-1);

   x = randn(d,x_2nd_dim);
   
   k_vecs[i] = floor(2.^(collect(linspace(4,m,15)))); # build our own logspace
   naive_times[i] = zeros(num_samples, length(k_vecs[i]));
   srht_times[i] = zeros(num_samples, length(k_vecs[i]));

   for j=1:length(k_vecs[i])
      k = k_vecs[i][j];
      @printf "\r m = %d, k = %d                             " m k
      for n=1:num_samples
         p = randperm(d)[1:k];
         #[p,~] = sort(p);

         time = @timed begin
            Hx = fht_C(x);
            #Hx = fht_ref(x);
            PHx = Hx[p,:];
         end
         naive_times[i][n,j] = time[2];

         time = @timed PHx = srht_C(x,p);
         #time = @timed PHx = srht_ref(x,p);
         srht_times[i][n,j] = time[2];
      end
   end
end
println()

# plot things
fig = figure(1);
ax = gca();
#ax = fig[:add_axes]([10; 100; 0; 1])
clf();
hold("on");
line1 = Void; line2 = Void; # scoping
for i=1:length(m_vec)
   #m = m_vec(i);
   #avg = mean(naive_times[i],1)
   #stddev = std(naive_times[i],0,1);
   #errorbar(k_vecs[i], avg, stddev, 'b--');
   
   avg = mean(naive_times[i][:]); # just plot average time as a constant, per Gunnar's recommendation
   line1, = plot([k_vecs[i][1]; k_vecs[i][end]], [avg; avg], "g--");

   m = m_vec[i];
   avg = mean(srht_times[i],1);
   stddev = std(srht_times[i],1);
   line2 = errorbar(vec(k_vecs[i]), vec(avg), yerr=vec(stddev), fmt="b-");
end

ax = gca()
ax[:set_xscale]("log");
xlabel("k - number of subsamples");
ylabel("runtime (s)");
title("Runtime scaling of SRHT vs. k (various d)");
fig[:legend]((line1, line2), ("Simple SRHT", "Better SRHT"), loc="lower right");
hold("off");
fig[:canvas][:draw]() # is this magic?
show()

end
# }}}

fht_test();
srht_test();
experiment_01();

