#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// use SIMD? (be sure to compile and link appropriately)
#define USE_SSE 1
#if USE_SSE
   #include <x86intrin.h> 
#endif

void split(unsigned *p,  unsigned len_p, unsigned val,
         unsigned *mp, unsigned **p1, unsigned *k1, unsigned **p2, unsigned *k2);


void srht_rec_had(double *pPHx, double *x, unsigned m, unsigned n, unsigned L,
         unsigned *pinds, unsigned len_p, unsigned len_inds_full, unsigned pstart)
{
// arrays coming from julia are column-major (Fortran) ordered

   static unsigned j,i;

   if ( L >= 4 ) {
      unsigned half, jhalf;
      unsigned mp, *p1, k1, *p2, k2;

      half = 2<<(L-2); // 2^(L-1)

      split(pinds, len_p, half-1, &mp, &p1, &k1, &p2, &k2);

      #if(USE_SSE)
      //TODO should we use posix_memalign?
      double *tmp = _mm_malloc(half*n*sizeof(*tmp), 16);
      #else
      double *tmp = malloc(half*n*sizeof(*tmp));
      #endif
      if ( tmp == NULL ) {
         printf("srht.c: memory error.\n"); exit(-1);
      }

      #if(USE_SSE) 
      // tmp = x1 + x2 with SSE2 intrinsics
      // half is always a factor of 4, so no cleanup
      // Julia arrays are 16 byte aligned (not 32), so must use AVX unaligned load, which
      // slows things down a bit.  
      static __m128d x1v, x2v, tmpv;
      for ( j=0; j<n; ++j ) {
         jhalf = j*half;
         for ( i=0; i<half; i+=2 ) {
            x1v = _mm_load_pd(x+2*jhalf+i); 
            x2v = _mm_load_pd(x+2*jhalf+i+half); 
            tmpv = _mm_add_pd(x1v, x2v);
            _mm_store_pd(tmp+jhalf+i, tmpv);
         }
      }

      #else
      // tmp = x1 + x2 with a simple loop
      for ( j=0; j<n; ++j ) {
         jhalf = j*half;
         for ( i=0; i<half; ++i ) { 
            tmp[jhalf+i] = x[2*jhalf+i] + x[2*jhalf+i+half];
         }
      }
      #endif
      
      if ( k1 > 0 ) {
         srht_rec_had(pPHx, tmp, half, n, L-1, p1, k1, len_inds_full, pstart);
      }
      
      #if(USE_SSE) 
      // tmp = x1 - x2 with SSE2 intrinsics
      // half is always a factor of 4, so no cleanup
      for ( j=0; j<n; ++j ) {
         jhalf = j*half;
         for ( i=0; i<half; i+=2 ) {
            x1v = _mm_load_pd(x+2*jhalf+i); 
            x2v = _mm_load_pd(x+2*jhalf+i+half); 
            tmpv = _mm_sub_pd(x1v, x2v);
            _mm_store_pd(tmp+jhalf+i, tmpv);
         }
      }

      #else
      // tmp = x1 - x2 with a simple loop
      for ( j=0; j<n; ++j ) {
         jhalf = j*half;
         for ( i=0; i<half; ++i ) { 
            tmp[jhalf+i] = x[2*jhalf+i] - x[2*jhalf+i+half];
         }
      }
      #endif
      
      if ( k2 > 0 ) {
         // modify p2 so indexing on next level down matches up with length of x1, x2
         for ( i=0; i<k2; ++i ) {
            p2[i] -= half;
         }

         srht_rec_had(pPHx, tmp, half, n, L-1, p2, k2, len_inds_full, pstart+k1);
      }

      free(p1); free(p2);
      
      #if(USE_SSE)
      _mm_free(tmp);
      #else
      free(tmp);
      #endif
   }

   else if ( L == 3 ) {
      static double x0,x1,x2,x3,x4,x5,x6,x7;
      static double Hx[8];
      static unsigned p_ptr, jm, jl;

      for ( j=0; j<n; ++j ) {
         jm = j*m;
         jl = j*len_inds_full;

         x0 = x[jm+0]; x1 = x[jm+1];
         x2 = x[jm+2]; x3 = x[jm+3];
         x4 = x[jm+4]; x5 = x[jm+5];
         x6 = x[jm+6]; x7 = x[jm+7];

         // transform values
         Hx[0] = x0+x1+x2+x3+x4+x5+x6+x7;
         Hx[1] = x0-x1+x2-x3+x4-x5+x6-x7;
         Hx[2] = x0+x1-x2-x3+x4+x5-x6-x7;
         Hx[3] = x0-x1-x2+x3+x4-x5-x6+x7;
         Hx[4] = x0+x1+x2+x3-x4-x5-x6-x7;
         Hx[5] = x0-x1+x2-x3-x4+x5-x6+x7;
         Hx[6] = x0+x1-x2-x3-x4-x5+x6+x7;
         Hx[7] = x0-x1-x2+x3-x4+x5+x6-x7;
         
         // subsample
         p_ptr = pstart;
         for ( i=0; i<len_p; ++i ) {
            pPHx[jl+p_ptr] = Hx[pinds[i]];
            p_ptr += 1;
         }
      }
   }

   else if ( L == 2 ) {
      static double x0, x1, x2, x3;
      static double Hx[4];
      static unsigned p_ptr, jm, jl;

      for ( j=0; j<n; ++j ) {
         jm = j*m;
         jl = j*len_inds_full;
         x0 = x[jm];
         x1 = x[jm+1];
         x2 = x[jm+2];
         x3 = x[jm+3];
         
         // transform values
         Hx[0] = x0+x1+x2+x3;
         Hx[1] = x0-x1+x2-x3;
         Hx[2] = x0+x1-x2-x3;
         Hx[3] = x0-x1-x2+x3;
         
         // subsample
         p_ptr = pstart;
         for ( i=0; i<len_p; ++i ) {
            pPHx[jl+p_ptr] = Hx[pinds[i]];
            p_ptr += 1;
         }
      }
   }

   else if ( L == 1 ) {
      static double x0, x1;
      static double Hx[2];
      static unsigned p_ptr, jm, jl;

      for ( j=0; j<n; ++j ) {
         jm = j*m;
         jl = j*len_inds_full;
         x0 = x[jm];
         x1 = x[jm+1];
         
         // transform values
         Hx[0] = x0+x1;
         Hx[1] = x0-x1;
         
         // subsample
         p_ptr = pstart;
         for ( i=0; i<len_p; ++i ) {
            pPHx[jl+p_ptr] = Hx[pinds[i]];
            p_ptr += 1;
         }
      }
   }

   else {
      printf("L went bad: L = %d\n", L);
      exit(-1);
   }

   return;
}

void split(unsigned *p,  unsigned len_p, unsigned val,
         unsigned *mp, unsigned **p1, unsigned *k1, unsigned **p2, unsigned *k2)
{
// Alg: https://en.wikipedia.org/wiki/Binary_search_algorithm

   unsigned L = 0, R = len_p-1, i;
      
   while ( 1 ) {
      if ( L > R ) {
         // binary search failed
         // assuming p is actually sorted, val may not exist in p
         // we'll go ahead and split at m, which should still split p appropriately
         *mp = R;
         break;
      }
   
      *mp = (unsigned)(floor((double)(L+R)/2.0));

      if ( p[*mp] < val ) {
         L = *mp + 1;
      }

      else if ( p[*mp] > val ) {
         if ( *mp == 0 ) break; // if len_p == 1, R can go negative (but unsigned)
         R = *mp - 1;

      }

      else { // p[*mp] == val
         break;
      }
   }

   if ( *mp == 0 && p[*mp] > val ) *k1 = 0;
   else *k1 = *mp+1;

   if ( *mp == len_p-1 && p[*mp] < val) *k2 = 0;
   else *k2 = len_p-*k1;
   
   *p1 = malloc(*k1*sizeof(*p1));
   if ( p1 == NULL ) {
      printf("srht.c: memory error.\n"); exit(-1);
   }

   for (i=0; i<*k1; ++i) {
      (*p1)[i] = p[i]; // note that *p1[i] == *(p1[i]), as all postfix operators 
                       // take precedence over all prefix operators
   }

   *p2 = malloc(*k2*sizeof(*p2));
   if ( p2 == NULL ) {
      printf("srht.c: memory error.\n"); exit(-1);
   }

   for (i=0; i<*k2; ++i) {
      (*p2)[i] = p[i+*k1];
   }
}

//void srht_rec_had(double *pPHx, double *x, unsigned m, unsigned n, 
//         unsigned start, unsigned stop, unsigned L, unsigned *pinds,
//         unsigned len_p, unsigned pstart, unsigned pstop)

// valgrind -v --track-origins=yes --leak-check=full ./main
int main(void) { // for valgrinding
   
   unsigned L = 14, n = 30;
   unsigned m = 2<<(L-1);
   double *px = malloc(m*n*sizeof(double));
   double *pPHx = malloc(m*n*sizeof(double));
   unsigned pinds[] = {0, 1, 4, 5};

   srht_rec_had(pPHx, px, m, n, L, pinds, 4, 4, 0);

   free(px);
   free(pPHx);
   
   return 1; 
}
