# julia> versioninfo()
# Julia Version 0.4.4
# Commit ae683af* (2016-03-15 15:43 UTC)
# Platform Info:
#   System: Linux (x86_64-unknown-linux-gnu)
#   CPU: Intel(R) Core(TM) i5-2467M CPU @ 1.60GHz
#   WORD_SIZE: 64
#   BLAS: libmkl_rt
#   LAPACK: libmkl_rt
#   LIBM: libimf
#   LLVM: libLLVM-3.3

include("fht.jl")

function srht_ref{T<:Number,IT<:Integer}(x::StridedArray{T}, p::Array{IT,1}, ordering::AbstractString="hadamard")
   if ndims(x) < 1 || ndims(x) > 2
      error("fht.jl: fht only works on vectors and matrices for now.");
   end
   
   (m,n) = size(x,1,2)

   if !ispow2(m)
      error(string("fht.jl: Invalid transform length.  Transform dimensions ",
         "should be powers of 2.  You can zero-pad to the next power of 2."))
   end
   L = Int64(round(log2(m)))
 
   p = p[:] # ensure p is a column vector
   I = sortperm(p) # sort p here for faster searching later
   inds = p[I]
   
   Iinv = zeros(eltype(p), length(I))
   Iinv[I] = 1:length(I) # inverse permutation
   
   PHx = srht_ref_rec(x, L, inds, ordering)
   PHx = PHx[Iinv,:]/sqrt(2)^L
   return PHx
end


function srht_ref_rec{T<:Number,IT<:Integer}(x::StridedArray{T},
      L::Integer, p::Array{IT,1}, ordering::AbstractString)

   if ordering == "hadamard" || ordering == "natural"

      if L >= 3
         half = 2<<(L-2) # 2^(L-1)
         m = searchsortedlast(p, half) # m == 0 or m == end+1 as edge cases
         p1 = p[1:m]; k1 = length(p1)
         p2 = p[m+1:end]; k2 = length(p2)
         # modify p2 so indexing on next level down matches up with length of x1, x2
         p2 -= half;
          
         x1 = x[1:half,:]
         x2 = x[half+1:2*half,:]
            
         PHx = zeros(T, k1+k2, size(x,2))
         if k1 > 0
            PHx[1:k1,:] = srht_ref_rec(x1+x2, L-1, p1, ordering)
         end

         if k2 > 0
            PHx[k1+1:k1+k2,:] = srht_ref_rec(x1-x2, L-1, p2, ordering)
         end

      elseif L == 2
         Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;
         PHx = Hx[p,:]

      elseif L == 1
         Hx = [1 1; 1 -1]*x;
         PHx = Hx[p,:]
         
      elseif L < 1
         error(string("srht.jl:srht_ref_rec: Invalid transform length.  Transform ",
            "dimensions should be positive powers of 2."))
      end

   elseif ordering == "sequency"
      error("srht.jl:srht_ref_rec: sequency order not implemented.")

   else
      error("srht.jl:srht_ref_rec: unsupported ordering: $(ordering)")

   end
   
   return PHx
end


function srht_test()
   
   #seed = 0;
   #println("seed = $(seed)");
   #srand(seed);

   #m = 3
   #x = Array{Float64}(collect(1:2^m))
   #p = [1; 2; 5; 6]
   
   m = 14
   x = randn(2^m, 30)
   k = rand(1:2^m)
   p = randperm(2^m)[1:k] # so this is one thing matlab does better...
   println([2^m k])

   @time begin 
      Hx = fht_ref(x)
      PHx = Hx[p,:]
   end
   
   @time PHx_ref = srht_ref(x, p)

   #display(PHx)
   #display(PHx_ref)

   @printf "norm(PHx - PHx_ref,2) = %e\n" norm(PHx - PHx_ref,2)

end
