
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// use SIMD? (be sure to compile and link appropriately)
#define USE_SSE 1
#if USE_SSE
   #include <x86intrin.h> 
#endif

void fht_rec_had(double *Hx, const double *x, const unsigned m, const unsigned n,
      unsigned start, unsigned stop, unsigned L)
{

   static unsigned j,jm;
   
   if ( L >= 3 ) { 
      static unsigned i;
      unsigned half, mid; // can't be static!
      
      half = 2<<(L-2); // 2^(L-1)
      mid = start+half-1;
      fht_rec_had(Hx, x, m, n, start, mid, L-1);
      fht_rec_had(Hx, x, m, n, mid+1, stop, L-1);

      #if(USE_SSE)
      // Loop with SSE2 intrinsics to do Hx1 <- Hx1 + Hx2
      //                                 Hx2 <- Hx1 - Hx2
      //
      // About 20% faster than the simple loop (m=2^16, n=30) with SSE2 (MMX)
      // Julia arrays are 16 byte aligned, so must use unaligned AVX load (slower than SSE2)
      static __m128d Hx1v,Hx2v, tmp1, tmp2;
      for (j = 0; j < n; ++j) {
         jm = j*m + start;
         
         // we know that half is divisible by 2, so no cleanup
         for (i=0; i < half; i+=2) {
            tmp1 = _mm_load_pd(Hx+jm+i);
            tmp2 = _mm_load_pd(Hx+jm+i+half);
            
            Hx1v = _mm_add_pd(tmp1, tmp2); // Hx1 + Hx2
            Hx2v = _mm_sub_pd(tmp1, tmp2); // Hx1 - Hx2
            
            _mm_store_pd(Hx+jm+i, Hx1v);
            _mm_store_pd(Hx+jm+i+half, Hx2v);
         }
      }
      
      #else
      // Simple loop to do Hx1 <- Hx1 + Hx2
      //                   Hx2 <- Hx1 - Hx2
      static double Hx1, Hx2;
      
      for (j = 0; j < n; ++j) {
         jm = j*m + start;
         for (i = 0; i < half; ++i) { 
            Hx1 = Hx[jm+i];
            Hx2 = Hx[jm+i+half];
            Hx[jm+i] = Hx1 + Hx2;
            Hx[jm+i+half] = Hx1 - Hx2;
         }
      }
      #endif

   }

   else if ( L == 2 ) {
      for (j = 0; j < n; ++j) {
         jm = j*m + start;
         static double x0,x1,x2,x3;
         x0 = x[jm+0]; x1 = x[jm+1];
         x2 = x[jm+2]; x3 = x[jm+3];

         Hx[jm+0] = x0 + x1 + x2 + x3;
         Hx[jm+1] = x0 - x1 + x2 - x3;
         Hx[jm+2] = x0 + x1 - x2 - x3;
         Hx[jm+3] = x0 - x1 - x2 + x3;
      }
   }

   else if ( L == 1 ) {
      for (j = 0; j < n; ++j) {
         jm = j*m + start;
         Hx[jm] = x[jm] + x[jm+1];
         Hx[jm+1] = x[jm] - x[jm+1];
      }
   }

   else {
      printf("L went bad!\n");
      exit(-1);
   }

}
