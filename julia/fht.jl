using ArrayViews
using Base.LinAlg.BLAS: axpy!

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


#TODO get sensible templating going on
#TODO should make a C+BLAS version
#function fht_rec!{T<:Number}(Hx::DenseArray{T}, x::DenseArray{T},
#      start::Integer, stop::Integer, L::Integer, ordering::AbstractString)
function fht_rec!{T<:Number}(pHx::Ptr{T}, px::Ptr{T}, m::Integer, n::Integer,
      start::Integer, stop::Integer, L::Integer, ordering::AbstractString)
   
   #(m,n) = size(x,1,2)
   
   if ordering == "hadamard" || ordering == "natural"

      if L >= 2
         # a more natural implementation
         #Hx = zeros(size(x))
         #half = 2<<(L-2) # 2^(L-1)
         #x1 = x[1:half,:]
         #x2 = x[half+1:2*half,:]
         #Hx1 = fht_rec(x1, L-1, ordering)
         #Hx2 = fht_rec(x2, L-1, ordering)
         #Hx[1:half,:] = Hx1 + Hx2
         #Hx[half+1:2*half,:] = Hx1 - Hx2
         
         #half = 2<<(L-2) # 2^(L-1)
         #mid = start+half-1
         #fht_rec!(Hx, x, start, mid, L-1, ordering)
         #fht_rec!(Hx, x, mid+1, stop, L-1, ordering)
         #Hx[start:mid,:] += Hx[mid+1:stop,:]
         ##axpy!(T(1), Hx[mid+1:stop,:], Hx[start:mid,:])
         #Hx[mid+1:stop,:] = Hx[start:mid,:] - 2*Hx[mid+1:stop,:]
         
         # doesn't work
         #half = 2<<(L-2) # 2^(L-1)
         #x1 = view(x, 1:half, :)
         #x2 = view(x, half+1:2*half, :)
         #Hx1 = view(Hx, 1:half, :)
         #Hx2 = view(Hx, half+1:2*half, :)
         #fht_rec!(Hx1, x1, 1, 1, L-1, ordering)
         #fht_rec!(Hx2, x2, 1, 1, L-1, ordering)
         #Hx1 += Hx2
         #Hx2 *= T(-2)
         #Hx2 += Hx1
         
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
         #Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x

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
         #Hx = [1 1; 1 -1]*x;
 
         @inbounds @simd for j in 0:n-1
            #Hx[start,j] = x[start,j] + x[stop,j]
            #Hx[stop,j] = x[start,j] - x[stop,j]
            jm = j*m
            unsafe_store!(pHx, unsafe_load(px, jm+start) + unsafe_load(px, jm+stop), jm+start)
            unsafe_store!(pHx, unsafe_load(px, jm+start) - unsafe_load(px, jm+stop), jm+stop)
         end
         
      elseif L < 1
         error(string("fht.jl:fht_rec: Invalide transform length.  Transform ",
            "dimensions should be positive powers of 2."))
      end

   elseif ordering == "sequency"
      error("fht.jl:fht_rec: sequency order not implemented.")

   else
      error("fht.jl:fht_rec:  unsupported ordering: $(ordering)")

   end
   
   #return Hx
end

# reference implementation used for testing (temporary)
# {{{
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
         error(string("fht.jl:fht_rec: Invalide transform length.  Transform ",
            "dimensions should be positive powers of 2."))
      end

   elseif ordering == "sequency"
      error("fht.jl:fht_rec: sequency order not implemented.")

   else
      error("fht.jl:fht_rec:  unsupported ordering: $(ordering)")

   end
   
   return Hx
end
# }}}

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
      
   m = 14
   #x = collect(1:2^m)
   #m = rand(1:16)
   x = randn(2^m,30)
   println(size(x))
   @time Hx = fht_ref(x)
   @time myHx = fht(x)
   
   #display(Hx)
   #display(myHx)
   @printf "norm(Hx - myHx,2) = %e" norm(Hx - myHx,2)

end


#vim: set noet:
