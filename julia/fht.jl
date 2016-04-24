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


"""
Hx = fht(x, ordering)

Naturally/Hadamard ordered fast Walsh-Hadamard transform
For matrices, we compute the transform over the columns
"""
function fht{T<:Number}(x::StridedArray{T}, ordering::AbstractString="hadamard")
  
   if ndims(x) < 1 || ndims(x) > 2
      error("fht.jl: fht only works on vectors and matrices for now.");
   end
   
   (m,n) = size(x,1,2)

   if !ispow2(m)
      error(string("fht.jl: Invalid transform length.  Transform dimensions ",
         "should be powers of 2.  You can zero-pad to the next power of 2."))
   end
   L = Int64(round(log2(m)))
   
   Hx = zeros(x)
   pHx = pointer(Hx)
   px = pointer(x)
   fht_rec!(pHx, px, m, n, 1, Int64(m), L, ordering)
   #fht_rec!(Hx, x, 1, Int64(m), L, ordering)
   Hx /= sqrt(2)^L
   
   return Hx
end


function fht_rec!{T<:Number}(pHx::Ptr{T}, px::Ptr{T}, m::Integer, n::Integer,
      start::Integer, stop::Integer, L::Integer, ordering::AbstractString)
   
   if ordering == "hadamard" || ordering == "natural"

      if L >= 2
         
         half = 2<<(L-2) # 2^(L-1)
         mid = start+half-1
         fht_rec!(pHx, px, m, n, start, mid, L-1, ordering)
         fht_rec!(pHx, px, m, n, mid+1, stop, L-1, ordering)
         
         for j in 0:n-1
            jm = j*m + start
            @inbounds @simd for i in 0:half-1
               Hx1 = unsafe_load(pHx, jm+i)
               Hx2 = unsafe_load(pHx, jm+i+half)
               unsafe_store!(pHx, Hx1 + Hx2, jm+i)
               unsafe_store!(pHx, Hx1 - Hx2, jm+i+half)
            end
         end
               
      elseif L == 2
         #Hx[start:stop,:] = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x[start:stop,:];

         @inbounds @simd for j in 0:n-1
            jm = j*m + start
            x1 = unsafe_load(px, jm + 0)
            x2 = unsafe_load(px, jm + 1)
            x3 = unsafe_load(px, jm + 2)
            x4 = unsafe_load(px, jm + 3)
            unsafe_store!(pHx, x1+x2+x3+x4, jm+0)
            unsafe_store!(pHx, x1-x2+x3-x4, jm+1)
            unsafe_store!(pHx, x1+x2-x3-x4, jm+2)
            unsafe_store!(pHx, x1-x2-x3+x4, jm+3)
         end

      elseif L == 1
         #Hx[start,:] = x[start,:] + x[stop,:]
         #Hx[stop,:] = x[start,:] - x[stop,:]
 
         @inbounds @simd for j in 0:n-1
            jm = j*m
            unsafe_store!(pHx, unsafe_load(px, jm+start) + unsafe_load(px, jm+stop), jm+start)
            unsafe_store!(pHx, unsafe_load(px, jm+start) - unsafe_load(px, jm+stop), jm+stop)
         end
         
      elseif L < 1
         error(string("fht.jl:fht_rec: Invalid transform length.  Transform ",
            "dimensions should be positive powers of 2."))
      end

   elseif ordering == "sequency"
      error("fht.jl:fht_rec: sequency order not implemented.")

   else
      error("fht.jl:fht_rec: unsupported ordering: $(ordering)")

   end
   
end

# Call C for recursive bit
function fht_C(x::DenseArray{Float64}, ordering::AbstractString="hadamard")

   if ndims(x) < 1 || ndims(x) > 2
      error("fht.jl: fht only works on vectors and matrices for now.");
   end
   
   (m,n) = size(x,1,2)

   if !ispow2(m)
      error(string("fht.jl: Invalid transform length.  Transform dimensions ",
         "should be powers of 2.  You can zero-pad to the next power of 2."))
   end
   L = Int64(round(log2(m)))
   
   Hx = zeros(x)
   if ordering == "hadamard" || ordering == "natural"
      pHx = pointer(Hx)
      px = pointer(x)
      um = UInt32(m)
      un = UInt32(n)
      ustart = UInt32(0) # these are C inds
      ustop = UInt32(m-1)
      uL = UInt32(L)
      
      ccall((:fht_rec_had, "./fht.so"), Void,
         (Ref{Float64}, Ref{Float64}, UInt32, UInt32, UInt32, UInt32, UInt32),
         pHx, px, um, un, ustart, ustop, uL)
      
   elseif ordering == "sequency"
      error("fht.jl:fht_rec: sequency order not implemented.")

   else
      error("fht.jl:fht_rec: unsupported ordering: $(ordering)")

   end
 
   Hx /= sqrt(2)^L

   return Hx
end


# reference implementation used for testing
# this is perhaps the most natural implementation of Ailon+Liberty, 2010.
function fht_ref{T<:Number}(x::StridedArray{T}, ordering::AbstractString="hadamard")
  
   if ndims(x) < 1 || ndims(x) > 2
      error("fht.jl: fht only works on vectors and matrices for now.");
   end
   
   (m,n) = size(x,1,2)

   if !ispow2(m)
      error(string("fht.jl: Invalid transform length.  Transform dimensions ",
         "should be powers of 2.  You can zero-pad to the next power of 2."))
   end
   L = Int64(round(log2(m)))
     
   Hx = fht_rec_ref(x, L, ordering)
   Hx /= sqrt(2)^L
   
   return Hx
end


function fht_rec_ref{T<:Number}(x::StridedArray{T}, L::Integer, ordering::AbstractString)
   
   if ordering == "hadamard" || ordering == "natural"

      if L >= 3
         Hx = zeros(size(x))
         half = 2<<(L-2) # 2^(L-1)
         x1 = x[1:half,:]
         x2 = x[half+1:2*half,:]
         Hx1 = fht_rec_ref(x1, L-1, ordering)
         Hx2 = fht_rec_ref(x2, L-1, ordering)
         Hx[1:half,:] = Hx1 + Hx2
         Hx[half+1:2*half,:] = Hx1 - Hx2

      elseif L == 2
         Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;

      elseif L == 1
         Hx = [1 1; 1 -1]*x;
         
      elseif L < 1
         error(string("fht.jl:fht_rec: Invalid transform length.  Transform ",
            "dimensions should be positive powers of 2."))
      end

   elseif ordering == "sequency"
      error("fht.jl:fht_rec: sequency order not implemented.")

   else
      error("fht.jl:fht_rec: unsupported ordering: $(ordering)")

   end
   
   return Hx
end


# A simple test function used for development
function fht_test()

   seed = 0;
   println("seed = $(seed)");
   srand(seed);

   # useful for development
	#m = 3
	#x = collect(1:2^m);
	#
	## natural order
	#Hx = [
	#   12.7279
	#   -1.4142
	#   -2.8284
	#         0
	#   -5.6569
	#         0
	#         0
	#         0];
	#
	#myHx = fht(x, "hadamard");

	#display(Hx)
   #display(myHx)
      
   m = 16
   #x = Array{Float64}(collect(1:2^m))
   #m = rand(1:16)
   x = randn(2^m,30)
   println(size(x))
   @time Hx = fht_ref(x)
   @time myHx = fht(x)
   @time Hx_C = fht_C(x)

   #display(Hx)
   #display(myHx)
   #display(Hx_C)
   @printf "norm(Hx - myHx,2) = %e\n" norm(Hx - myHx,2)
   @printf "norm(Hx - Hx_C,2) = %e\n" norm(Hx - Hx_C,2)

end

function fht_timing()
   seed = 0;
   println("seed = $(seed)");
   srand(seed);

   m = 20
   x = randn(2^m,1)
   println(size(x))
   
   n_samples = 10
   t_ref = 0.0
   t = 0.0
   t_C = 0.0
   for i=1:n_samples
      time = @timed Hx = fht_ref(x)
      t_ref += time[2];
      time = @timed myHx = fht(x)
      t += time[2];
      time = @timed Hx_C = fht_C(x)
      t_C += time[2];
   end

   @printf "<t_ref> = %f\n" t_ref / n_samples
   @printf "<t>     = %f\n" t / n_samples
   @printf "<t_C>   = %f\n" t_C / n_samples

end

#vim: set noet:

