

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// use SIMD? (be sure to compile and link appropriately)
#define USE_SIMD 1
#if USE_SIMD
   #include <x86intrin.h> 
#endif

void fht_rec_had(double *Hx, const double *x, const unsigned m, const unsigned n,
      unsigned start, unsigned stop, unsigned L)
{

   unsigned j,jm;
   
   if ( L >= 3 ) { 
      double Hx1, Hx2;
      unsigned i, half, mid;
      
      half = 2<<(L-2); // 2^(L-1)
      mid = start+half-1;
      fht_rec_had(Hx, x, m, n, start, mid, L-1);
      fht_rec_had(Hx, x, m, n, mid+1, stop, L-1);

      // Loop with SIMD to do Hx1 <- Hx1 + Hx2
      //                      Hx2 <- Hx1 - Hx2
      if (USE_SIMD) {
         __m128d Hx1v,Hx2v, tmp1, tmp2;

         __m128d neg_one;
         neg_one = _mm_set_pd1(-1.0);

         for (j = 0; j < n; ++j) {
            jm = j*m + start;
            //Hx_jm = Hx + j*m + start; //TODO test aliasing to make it faster?
            
            // we know that half is divisible by 2
            for (i=0; i < half; i+=2) {
               tmp1 = _mm_load_pd(Hx+jm+i);
               tmp2 = _mm_load_pd(Hx+jm+i+half);
               
               Hx1v = _mm_add_pd(tmp1, tmp2); // Hx1 + Hx2
               tmp2 = _mm_mul_pd(neg_one, tmp2);
               Hx2v = _mm_add_pd(tmp1, tmp2); // Hx1 - Hx2
               
               _mm_store_pd(Hx+jm+i, Hx1v);
               _mm_store_pd(Hx+jm+i+half, Hx2v);
            }
         }
      }

      // Simple loop to do Hx1 <- Hx1 + Hx2
      //                   Hx2 <- Hx1 - Hx2
      else {
         for (j = 0; j < n; ++j) {
            jm = j*m + start;
            for (i = 0; i < half; ++i) { 
               Hx1 = Hx[jm+i];
               Hx2 = Hx[jm+i+half];
               Hx[jm+i] = Hx1 + Hx2;
               Hx[jm+i+half] = Hx1 - Hx2;
            }
         }
      } 
   }

   else if ( L == 2 ) {
      double x0,x1,x2,x3;
      for (j = 0; j < n; ++j) {
         jm = j*m + start;
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
