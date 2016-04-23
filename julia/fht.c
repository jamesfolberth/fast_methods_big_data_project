

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// use MKL? (be sure to compile and link appropriately)
#define USE_MKL 0
#if USE_MKL
   #include "mkl.h"
#endif

//// use SIMD? (be sure to compile and link appropriately)
//#define USE_SIMD 1
//#if USE_SIMD
//#endif

void fht_rec_had_1(double *Hx, const double *x, const unsigned m, const unsigned n,
      unsigned start, unsigned stop, unsigned L)
{

   unsigned j,jm;
   
   if ( L >= 3 ) { 
      double Hx1, Hx2;
      unsigned i, half, mid;
      
      half = 2<<(L-2); // 2^(L-1)
      mid = start+half-1;
      fht_rec_had_1(Hx, x, m, n, start, mid, L-1);
      fht_rec_had_1(Hx, x, m, n, mid+1, stop, L-1);
      
      if (USE_MKL) {
         // Probably won't gain much from MKL here... 

         // A bogus test call to MKL's BLAS
         //int N = m*n, incx = 1;
         //double alpha = 2;
         //dscal(&N, &alpha, Hx, &incx);

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
