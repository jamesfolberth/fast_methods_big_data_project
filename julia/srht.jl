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

#using ProfileView

#include("fht.jl")

SRHT_SWITCH_LEVEL = -1

function srht{T<:Number,IT<:Integer}(x::StridedArray{T}, p::Array{IT,1}, ordering::AbstractString="hadamard")
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
   
   Iinv = Array{IT}(length(I))
   Iinv[I] = 1:length(I) # inverse permutation
   
   PHx = Array{T}(length(I), n)
   pPHx = pointer(PHx)
   px = pointer(x)
   #srht_rec!(pPHx, px, m, n, 1, m, L, inds, 1, length(inds), ordering)
   srht_rec!(PHx, x, m, n, 1, m, L, inds, 1, length(inds), ordering)

   #if L >= SRHT_SWITCH_LEVEL # srht_rec! index across rows a lot at lower levels.
   #   PHx = Array{T}(length(I), n)
   #   srht_rec!(PHx, x, m, n, 1, m, L, inds, 1, length(inds), ordering)
   #else
   #   PHx = srht_ref_rec(x, L, inds, ordering)
   #end
   
   #PHx = Array{T}(length(I), n)
   #for j=1:n
   #   srht_rec!(PHx, x[:,j], j, m, n, 1, m, L, inds, 1, length(inds), ordering)
   #end


   PHx = PHx[Iinv,:]/sqrt(2)^L
   return PHx
end

# {{{ srht_rec! with pointers (and bugs...)
function srht_rec!{T<:Number,IT<:Integer}(pPHx::Ptr{T}, x::Ptr{T},
      m::Integer, n::Integer, start::Integer, stop::Integer, L::Integer,
      p::Array{IT,1}, pstart::Integer, pstop::Integer, ordering::AbstractString)
   
   if ordering == "hadamard" || ordering == "natural"

      if L >= 3
         half = 2<<(L-2) # 2^(L-1)
         mid = start+half-1

         #mp = searchsortedlast(p, half) # m == 0 or m == end+1 as edge cases
         mp = 1
         p1 = p[1:mp]; k1 = length(p1)
         p2 = p[mp+1:end]; k2 = length(p2)
         # modify p2 so indexing on next level down matches up with length of x1, x2
         p2 -= half;
                   
         tmp = Array{T}(half, n)
         pt = pointer(tmp)
         for j in 0:n-1
            jm = j*m
            for i in 0:half-1
               #tmp[i,j] = x[i,j] + x[i+half,j] # x1+x2
               #x1 = unsafe_load(px, jm+i)
               #x2 = unsafe_load(px, jm+i+half)
               #x1 = 0.0; x2 = 0.0;
               #unsafe_store!(pt, x1+x2, jm+i)
            end
         end

         if k1 > 0
            #tmp = x[1:half,:] + x[half+1:2*half,:] # x1 + x2
            #PHx[1:k1,:] = srht_rec!(PHx, tmp, L-1, p1, ordering)
            
            srht_rec!(pPHx, pt, m, n, start, mid, L-1, p1, pstart, pstart+k1-1, ordering)
         end
         
         #copy!(tmp, x[1:half,:])
         #for j in 0:n-1
         #   jm = j*m
         #   for i in 0:half-1
         #      #tmp[i,j] = x[i,j] + x[i+half,j] # x1+x2
         #      x1 = unsafe_load(px, jm+i)
         #      x2 = unsafe_load(px, jm+i+half)
         #      unsafe_store!(pt, x1-x2, jm+i)
         #   end
         #end

         if k2 > 0
            #tmp = x[1:half,:] - x[half+1:2*half,:] # x1 - x2
            #PHx[k1+1:k1+k2,:] = srht_rec(tmp, L-1, p2, ordering)
            
            srht_rec!(pPHx, pt, m, n, mid+1, stop, L-1, p2, pstart+k1, pstop, ordering)
         end

      elseif L == 2
         #Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;
         #PHx[pstart:pstop,:] = Hx[p,:]
         
         #I = pstop-pstart
         #@inbounds for j=0:n-1
         #   jm = j*m + pstart
         #   for i = 0:I
         #      unsafe_store!(pPHx, Hx[p[i+1],j+1], jm + i)
         #   end
         #end

         #@simd for j in 0:n-1
			#	jm = j*m
         #   x1 = unsafe_load(px, jm + 0)
         #   x2 = unsafe_load(px, jm + 1)
         #   x3 = unsafe_load(px, jm + 2)
         #   x4 = unsafe_load(px, jm + 3)
			#			
			#	p_ptr = pstart
			#	for i in 1:length(p) # I don't like all of these decisions in an inner loop...
			#		if p[i] == 1
			#			unsafe_store!(pPHx, x1+x2+x3+x4, jm+p_ptr)
			#			p_ptr += 1
         #      elseif p[i] == 2
			#			unsafe_store!(pPHx, x1-x2+x3-x4, jm+p_ptr)
			#			p_ptr += 1
         #      elseif p[i] == 3
			#			unsafe_store!(pPHx, x1+x2-x3-x4, jm+p_ptr)
			#			p_ptr += 1
         #      else
			#			unsafe_store!(pPHx, x1-x2-x3+x4, jm+p_ptr)
			#			p_ptr += 1
         #      end
			#	end
         #end


      elseif L == 1
			#TODO
         #Hx = [1 1; 1 -1]*x;
         #PHx[pstart:pstop,:] = Hx[p,:]
         
      elseif L < 1
         error(string("srht.jl:srht_ref_rec: Invalid transform length.  Transform ",
            "dimensions should be positive powers of 2."))
      end

   elseif ordering == "sequency"
      error("srht.jl:srht_ref_rec: sequency order not implemented.")

   else
      error("srht.jl:srht_ref_rec: unsupported ordering: $(ordering)")

   end
   
end
# }}}

## {{{ srht_rec! on a single column
#function srht_rec!{T<:Number,IT<:Integer}(PHx::StridedArray{T}, x::StridedArray{T}, J::Integer,
#      m::Integer, n::Integer, start::Integer, stop::Integer, L::Integer,
#      p::Array{IT,1}, pstart::Integer, pstop::Integer, ordering::AbstractString)
#   
#   if ordering == "hadamard" || ordering == "natural"
#
#      if L >= 3
#         half = 2<<(L-2) # 2^(L-1)
#         mid = start+half-1
#
#         m = searchsortedlast(p, half) # m == 0 or m == end+1 as edge cases
#         p1 = p[1:m]; k1 = length(p1)
#         p2 = p[m+1:end]; k2 = length(p2)
#         # modify p2 so indexing on next level down matches up with length of x1, x2
#         p2 -= half;
#                   
#         tmp = Array{eltype(x)}(half, 1)
#         @inbounds @simd for i in 1:half
#            tmp[i] = x[i] + x[i+half] # x1+x2
#         end
#
#         if k1 > 0
#            #tmp = x[1:half,:] + x[half+1:2*half,:] # x1 + x2
#            #PHx[1:k1,:] = srht_rec!(PHx, tmp, L-1, p1, ordering)
#            
#            srht_rec!(PHx, tmp, J, m, n, start, mid, L-1, p1, pstart, pstart+k1-1, ordering)
#         end
#         
#         #copy!(tmp, x[1:half,:])
#         @inbounds @simd for i in 1:half
#            tmp[i] = x[i] - x[i+half] # x1-x2
#         end
#
#         if k2 > 0
#            #tmp = x[1:half,:] - x[half+1:2*half,:] # x1 - x2
#            #PHx[k1+1:k1+k2,:] = srht_rec(tmp, L-1, p2, ordering)
#            
#            srht_rec!(PHx, tmp, J, m, n, mid+1, stop, L-1, p2, pstart+k1, pstop, ordering)
#         end
#
#      elseif L == 2
#         Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;
#         PHx[pstart:pstop,J] = Hx[p]
#
#      elseif L == 1
#         Hx = [1 1; 1 -1]*x;
#         PHx[pstart:pstop,J] = Hx[p]
#         
#      elseif L < 1
#         error(string("srht.jl:srht_ref_rec: Invalid transform length.  Transform ",
#            "dimensions should be positive powers of 2."))
#      end
#
#   elseif ordering == "sequency"
#      error("srht.jl:srht_ref_rec: sequency order not implemented.")
#
#   else
#      error("srht.jl:srht_ref_rec: unsupported ordering: $(ordering)")
#
#   end
#   
#end
## }}}

# {{{ srht_rec! with SRHT_SWITCH_LEVEL
#function srht_rec!{T<:Number,IT<:Integer}(PHx::StridedArray{T}, x::StridedArray{T},
#      m::Integer, n::Integer, start::Integer, stop::Integer, L::Integer,
#      p::Array{IT,1}, pstart::Integer, pstop::Integer, ordering::AbstractString)
#   
#   if ordering == "hadamard" || ordering == "natural"
#
#      if L >= SRHT_SWITCH_LEVEL && L >= 3
#         half = 2<<(L-2) # 2^(L-1)
#         mid = start+half-1
#
#         m = searchsortedlast(p, half) # m == 0 or m == end+1 as edge cases
#         p1 = p[1:m]; k1 = length(p1)
#         p2 = p[m+1:end]; k2 = length(p2)
#         # modify p2 so indexing on next level down matches up with length of x1, x2
#         p2 -= half;
#                   
#         #tmp = Array{eltype(x)}(half, n)
#         #@inbounds for j in 1:n
#         #   jm = j*m
#         #   @simd for i in 1:half
#         #      tmp[i,j] = x[i,j] + x[i+half,j] # x1+x2
#         #      #tmp[jm+i] = x[jm+i] + x[jm+i+half] # x1+x2
#         #   end
#         #end
#         x1 = x[1:half,:]
#         x2 = x[half+1:2*half,:]
#
#         if k1 > 0
#            #tmp = x[1:half,:] + x[half+1:2*half,:] # x1 + x2
#            #PHx[1:k1,:] = srht_rec!(PHx, tmp, L-1, p1, ordering)
#            
#            #srht_rec!(PHx, tmp, m, n, start, mid, L-1, p1, pstart, pstart+k1-1, ordering)
#            srht_rec!(PHx, x1+x2, m, n, start, mid, L-1, p1, pstart, pstart+k1-1, ordering)
#         end
#         
#         #copy!(tmp, x[1:half,:])
#         #@inbounds for j in 1:n
#         #   jm = j*m
#         #   @simd for i in 1:half
#         #      tmp[i,j] = x[i,j] - x[i+half,j] # x1-x2
#         #      #tmp[jm+i] = x[jm+i] - x[jm+i+half] # x1+x2
#         #   end
#         #end
#
#         if k2 > 0
#            #tmp = x[1:half,:] - x[half+1:2*half,:] # x1 - x2
#            #PHx[k1+1:k1+k2,:] = srht_rec(tmp, L-1, p2, ordering)
#            
#            #srht_rec!(PHx, tmp, m, n, mid+1, stop, L-1, p2, pstart+k1, pstop, ordering)
#            srht_rec!(PHx, x1-x2, m, n, mid+1, stop, L-1, p2, pstart+k1, pstop, ordering)
#         end
#
#      elseif L == 2
#         Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;
#         PHx[pstart:pstop,:] = Hx[p,:]
#
#      elseif L == 1
#         Hx = [1 1; 1 -1]*x;
#         PHx[pstart:pstop,:] = Hx[p,:]
#      
#      else
#         println("I'm here")
#         PHx[pstart:pstop,:] = srht_ref_rec(x, L, p, ordering)
#      end
#
#   elseif ordering == "sequency"
#      error("srht.jl:srht_ref_rec: sequency order not implemented.")
#
#   else
#      error("srht.jl:srht_ref_rec: unsupported ordering: $(ordering)")
#
#   end
#   
#end
# }}}

## {{{ srht_rec!
function srht_rec!{T<:Number,IT<:Integer}(PHx::StridedArray{T}, x::StridedArray{T},
      m::Integer, n::Integer, start::Integer, stop::Integer, L::Integer,
      p::Array{IT,1}, pstart::Integer, pstop::Integer, ordering::AbstractString)
   
   if ordering == "hadamard" || ordering == "natural"

      if L >= 3
         half = 2<<(L-2) # 2^(L-1)
         mid = start+half-1

         m = searchsortedlast(p, half) # m == 0 or m == end+1 as edge cases
         p1 = p[1:m]; k1 = length(p1)
         p2 = p[m+1:end]; k2 = length(p2)
         # modify p2 so indexing on next level down matches up with length of x1, x2
         p2 -= half;
            
         x1 = x[1:half,:]
         x2 = x[half+1:2*half,:]
         #tmp = x1+x2
         #tmp = Array{T}(half, n)
         #tmp = x[1:half,:] + x[half+1:2*half,:]
         #@inbounds for j in 1:n
         #   jm = j*m
         #   @simd for i in 1:half
         #      tmp[i,j] = x[i,j] + x[i+half,j] # x1+x2
         #      #tmp[jm+i] = x[jm+i] + x[jm+i+half] # x1+x2
         #   end
         #end
         #println(x1+x2)

         if k1 > 0
            #tmp = x[1:half,:] + x[half+1:2*half,:] # x1 + x2
            #PHx[1:k1,:] = srht_rec!(PHx, tmp, L-1, p1, ordering)
            
            srht_rec!(PHx, x1+x2, m, n, start, mid, L-1, p1, pstart, pstart+k1-1, ordering)
         end
         
         #copy!(tmp, x[1:half,:])
         #@inbounds for j in 1:n
         #   jm = j*m
         #   @simd for i in 1:half
         #      tmp[i,j] = x[i,j] - x[i+half,j] # x1-x2
         #      #tmp[jm+i] = x[jm+i] - x[jm+i+half] # x1+x2
         #   end
         #end
         #tmp = x[1:half,:] - x[half+1:2*half,:]
         #tmp = x1-x2

         if k2 > 0
            #tmp = x[1:half,:] - x[half+1:2*half,:] # x1 - x2
            #PHx[k1+1:k1+k2,:] = srht_rec(tmp, L-1, p2, ordering)
            
            srht_rec!(PHx, x1-x2, m, n, mid+1, stop, L-1, p2, pstart+k1, pstop, ordering)
         end

      #elseif L == 4
      #   Hx = [ 1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; 
      #          1  -1   1  -1   1  -1   1  -1   1  -1   1  -1   1  -1   1  -1;
      #          1   1  -1  -1   1   1  -1  -1   1   1  -1  -1   1   1  -1  -1;
      #          1  -1  -1   1   1  -1  -1   1   1  -1  -1   1   1  -1  -1   1;
      #          1   1   1   1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1;
      #          1  -1   1  -1  -1   1  -1   1   1  -1   1  -1  -1   1  -1   1;
      #          1   1  -1  -1  -1  -1   1   1   1   1  -1  -1  -1  -1   1   1;
      #          1  -1  -1   1  -1   1   1  -1   1  -1  -1   1  -1   1   1  -1;
      #          1   1   1   1   1   1   1   1  -1  -1  -1  -1  -1  -1  -1  -1;
      #          1  -1   1  -1   1  -1   1  -1  -1   1  -1   1  -1   1  -1   1;
      #          1   1  -1  -1   1   1  -1  -1  -1  -1   1   1  -1  -1   1   1;
      #          1  -1  -1   1   1  -1  -1   1  -1   1   1  -1  -1   1   1  -1;
      #          1   1   1   1  -1  -1  -1  -1  -1  -1  -1  -1   1   1   1   1;
      #          1  -1   1  -1  -1   1  -1   1  -1   1  -1   1   1  -1   1  -1;
      #          1   1  -1  -1  -1  -1   1   1  -1  -1   1   1   1   1  -1  -1;
      #          1  -1  -1   1  -1   1   1  -1  -1   1   1  -1   1  -1  -1   1]*x;
      #   PHx[pstart:pstop,:] = Hx[p,:]
      #
      #elseif L == 3
      #   Hx = [1  1  1  1  1  1  1  1; 
		#			1 -1  1 -1  1 -1  1 -1;
		#			1  1 -1 -1  1  1 -1 -1;
		#			1 -1 -1  1  1 -1 -1  1;
		#			1  1  1  1 -1 -1 -1 -1;
		#			1 -1  1 -1 -1  1 -1  1;
		#			1  1 -1 -1 -1 -1  1  1;
		#			1 -1 -1  1 -1  1  1 -1]*x
      #   PHx[pstart:pstop,:] = Hx[p,:]

      elseif L == 2
         Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;
         PHx[pstart:pstop,:] = Hx[p,:]

      elseif L == 1
         Hx = [1 1; 1 -1]*x;
         PHx[pstart:pstop,:] = Hx[p,:]
         
      elseif L < 1
         error(string("srht.jl:srht_ref_rec: Invalid transform length.  Transform ",
            "dimensions should be positive powers of 2."))
      end

   elseif ordering == "sequency"
      error("srht.jl:srht_ref_rec: sequency order not implemented.")

   else
      error("srht.jl:srht_ref_rec: unsupported ordering: $(ordering)")

   end
   
end
## }}}


## {{{ srht2
#function srht2{T<:Number,IT<:Integer}(x::StridedArray{T}, p::Array{IT,1}, ordering::AbstractString="hadamard")
#   if ndims(x) < 1 || ndims(x) > 2
#      error("fht.jl: fht only works on vectors and matrices for now.");
#   end
#   
#   (m,n) = size(x,1,2)
#
#   if !ispow2(m)
#      error(string("fht.jl: Invalid transform length.  Transform dimensions ",
#         "should be powers of 2.  You can zero-pad to the next power of 2."))
#   end
#   L = Int64(round(log2(m)))
# 
#   p = p[:] # ensure p is a column vector
#   I = sortperm(p) # sort p here for faster searching later
#   inds = p[I]
#   
#   Iinv = zeros(eltype(p), length(I))
#   Iinv[I] = 1:length(I) # inverse permutation
#   
#   PHx = Array{T}(length(I),n)
#   srht_rec2!(PHx, x, L, inds, 1, m, ordering)
#   PHx = PHx[Iinv,:]/sqrt(2)^L
#   return PHx
#end
#
#
#function srht_rec2!{T<:Number,IT<:Integer}(PHx, x::StridedArray{T},
#      L::Integer, p::Array{IT,1}, pstart, pstop, ordering::AbstractString)
#
#   if ordering == "hadamard" || ordering == "natural"
#
#      if L >= 3
#         half = 2<<(L-2) # 2^(L-1)
#         m = searchsortedlast(p, half) # m == 0 or m == end+1 as edge cases
#         p1 = p[1:m]; k1 = length(p1)
#         p2 = p[m+1:end]; k2 = length(p2)
#         # modify p2 so indexing on next level down matches up with length of x1, x2
#         p2 -= half;
#          
#         x1 = x[1:half,:]
#         x2 = x[half+1:2*half,:]
#            
#         PHx = zeros(T, k1+k2, size(x,2))
#         if k1 > 0
#            #PHx[1:k1,:] = srht_rec2!(x1+x2, L-1, p1, ordering)
#            srht_rec2!(PHx, x1+x2, L-1, p1, pstart, pstart+k1-1, ordering)
#         end
#
#         if k2 > 0
#            #PHx[k1+1:k1+k2,:] = srht_rec2!(x1-x2, L-1, p2, ordering)
#            srht_rec2!(PHx, x1-x2, L-1, p2, pstart+k1, pstop, ordering)
#         end
#
#      elseif L == 2
#         Hx = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]*x;
#         #PHx = Hx[p,:]
#         PHx[pstart:pstop,:] = Hx[p,:]
#
#      elseif L == 1
#         Hx = [1 1; 1 -1]*x;
#         #PHx = Hx[p,:]
#         PHx[pstart:pstop,:] = Hx[p,:]
#         
#      elseif L < 1
#         error(string("srht.jl:srht_ref_rec: Invalid transform length.  Transform ",
#            "dimensions should be positive powers of 2."))
#      end
#
#   elseif ordering == "sequency"
#      error("srht.jl:srht_ref_rec: sequency order not implemented.")
#
#   else
#      error("srht.jl:srht_ref_rec: unsupported ordering: $(ordering)")
#
#   end
#   
#   return PHx
#end
## }}}


function srht_C{T<:Float64,IT<:Integer}(x::StridedArray{T}, p::Array{IT,1}, ordering::AbstractString="hadamard")
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
   inds = Array{UInt32}(p[I])
   
   Iinv = Array{UInt32}(length(I))
   Iinv[I] = 1:length(I) # inverse permutation

   
   PHx = Array{T}(length(I), n)
   if ordering == "hadamard" || ordering == "natural"
      pPHx = pointer(PHx)
      px = pointer(x)
      um = UInt32(m)
      un = UInt32(n)
      #ustart = UInt32(0) #  C index
      #ustop = UInt32(m-1) # C index
      uL = UInt32(L)
      inds -= 1; # C indexes
      pinds = pointer(inds)
      len_p = UInt32(length(inds))
      len_inds_full = UInt32(length(inds))
      pstart = UInt32(0) # C index
      #pstop = UInt32(length(inds)-1) # C index

      #srht_rec!(pPHx, px, m, n, 1, m, L, inds, 1, length(inds), ordering)
      ccall((:srht_rec_had, "./srht.so"), Void,
         (Ref{T}, Ref{T}, UInt32, UInt32, UInt32,
          Ref{UInt32}, UInt32, UInt32, UInt32),
         pPHx, px, um, un, uL, pinds, len_p, len_inds_full, pstart)

   end


   PHx = PHx[Iinv,:]/sqrt(2)^L
   return PHx
end


# Reference implementation
# {{{
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
# }}}


function srht_test()
   
   include("fht.jl")

   seed = 0;
   println("seed = $(seed)");
   srand(seed);

   #m = 3
   #x = Array{Float64}(collect(1:2^m))
   #p = [1; 2; 5; 6]
   ##k = 5; p = randperm(2^m)[1:k] # so this is one thing matlab does better...
   
   m = 14
   x = randn(2^m, 300)
   k = rand(1:2^m)
   p = randperm(2^m)[1:k] # so this is one thing matlab does better...
   println([2^m k])

   @time begin 
      Hx = fht_C(x)
      PHx_naive = Hx[p,:]
   end
   
   @time PHx_ref = srht_ref(x, p)
   @time PHx = srht(x, p)
   @time PHx_C = srht_C(x, p)
   #Profile.clear()
   #@profile PHx = srht(x, p)

   #display(PHx_naive)
   #display(PHx_ref)
   #display(PHx)
   #display(PHx_C)

   @printf "norm(PHx_naive - PHx_ref,2) = %e\n" vecnorm(PHx_naive - PHx_ref)
   @printf "norm(PHx_naive - PHx,2) = %e\n" vecnorm(PHx_naive - PHx)
   @printf "norm(PHx_naive - PHx_C,2) = %e\n" vecnorm(PHx_naive - PHx_C)

end

function srht_timing()
   seed = 0;
   println("seed = $(seed)");
   srand(seed);

   m = 14
   x = randn(2^m, 30)
   k = rand(1:2^m)
   p = randperm(2^m)[1:k] # so this is one thing matlab does better...
   println([2^m k])

   n_samples = 50
   t_fht_C = zeros(n_samples,1);
   t_ref = zeros(n_samples,1);
   t_C = zeros(n_samples,1);
   for i=1:n_samples
      time = @timed begin
         Hx = fht_C(x)
         PHx = Hx[p,:]
      end
      t_fht_C[i] = time[2];

      time = @timed PHx_ref = srht_ref(x, p)
      t_ref[i] = time[2];

      time = @timed PHx_C = srht_C(x, p)
      t_C[i] = time[2];
   end

   @printf "<t_fht_C> = %f +- %f\n" mean(t_fht_C) std(t_fht_C)
   @printf "<t_ref>   = %f +- %f\n" mean(t_ref) std(t_ref)
   @printf "<t_C>     = %f +- %f\n" mean(t_C) std(t_C)

end


