#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// use SIMD? (be sure to compile and link appropriately)
#define USE_SIMD 0
#if USE_SIMD
   #include <x86intrin.h> 
#endif

void split(unsigned *p,  unsigned len_p, unsigned val,
         unsigned *mp, unsigned **p1, unsigned *k1, unsigned **p2, unsigned *k2);


void srht_rec_had(double *pPHx, double *x, unsigned m, unsigned n, unsigned L,
         unsigned *pinds, unsigned len_p, unsigned len_inds_full, unsigned pstart)
{
// arrays coming from julia are column-major (Fortran) ordered

   static unsigned j,i;

   if ( L >= 3 ) {
      unsigned half, jhalf;
      unsigned mp, *p1, k1, *p2, k2;

      half = 2<<(L-2); // 2^(L-1)

      split(pinds, len_p, half-1, &mp, &p1, &k1, &p2, &k2);

      //printf("k1 = %d, k2 = %d\n", k1, k2);

      double *tmp = malloc(half*n*sizeof(double));
      if ( tmp == NULL ) {
         printf("srht.c: memory error.\n"); exit(-1);
      }
      
      //TODO (inline) function?
      #if(USE_SIMD) 
      // tmp = x1 + x2 with SIMD
      // half is always a factor of 2 (or 4), so no cleanup
      printf("SIMD NOT IMPLEMENTED\n");
      exit(-1);

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
      
      //TODO (inline) function?
      #if(USE_SIMD) 
      // tmp = x1 - x2 with SIMD
      // half is always a factor of 2 (or 4), so no cleanup
      printf("SIMD NOT IMPLEMENTED\n");
      exit(-1);

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

      free(p1);
      free(p2);
      free(tmp);
   }

   else if ( L == 2 ) {
      static double x0, x1, x2, x3; //TODO profile the static part
      static unsigned p_ptr, jm, jl;

      for ( j=0; j<n; ++j ) {
         jm = j*m;
         jl = j*len_inds_full;
         x0 = x[jm];
         x1 = x[jm+1];
         x2 = x[jm+2];
         x3 = x[jm+3];
         
         p_ptr = pstart;
         //printf("len_p = %d\n", len_p);
         for ( i=0; i < len_p; ++i ) { //TODO lots of decisions in this inner loop...
                                       //     Set x0,...,x3 equal to transform values, then
                                       //     play p_ptr games to add in the right one to the
                                       //     right spot?
            //printf("p_ptr = %d, len_p = %d\n", p_ptr, len_p);
            //printf("pinds[i] = %d\n", pinds[i]); 
            
            //printf("jl+p_ptr = %d\n", jl+p_ptr);

            if ( pinds[i] == 0 ) {
               pPHx[jl+p_ptr] = x0+x1+x2+x3;
               //printf("I'm here: 0\n");
               p_ptr += 1;
            }

            else if ( pinds[i] == 1 ) {
               pPHx[jl+p_ptr] = x0-x1+x2-x3;
               //printf("I'm here: 1\n");
               p_ptr += 1;
            }

            else if ( pinds[i] == 2 ) {
               pPHx[jl+p_ptr] = x0+x1-x2-x3;
               //printf("I'm here: 2\n");
               p_ptr += 1;
            }

            else if ( pinds[i] == 3 ) {
               pPHx[jl+p_ptr] = x0-x1-x2+x3;
               //printf("I'm here: 3\n");
               p_ptr += 1;
            }
         }
      }
   }

   else if ( L == 1 ) {
      printf("L = 1 NOT IMPLEMENTED\n");
      exit(-1);
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
      
   //printf("val = %d\n", val);
   //printf("len_p = %d\n", len_p);
   //for ( i=0; i<len_p; ++i ) {
   //   printf("p[i] = %d\n", p[i]);
   //}

   while ( 1 ) {
      if ( L > R ) {
         // binary search failed
         // assuming p is actually sorted, val may not exist in p
         // we'll go ahead and split at m, which should still split p appropriately
         *mp = R;
         //printf("bsearch failed\n");
         break;
      }
   
      *mp = (unsigned)(floor((double)(L+R)/2.0));
      //printf("L,R,*mp = %d, %d, %d\n", L,R,*mp);

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
   //printf("L = %d, *mp = %d, R = %d\n", L, *mp, R);

   if ( *mp == 0 && p[*mp] > val ) *k1 = 0;
   else *k1 = *mp+1;

   if ( *mp == len_p-1 && p[*mp] < val) *k2 = 0;
   else *k2 = len_p-*k1;
   
   //*k1 = *mp+1;
   *p1 = malloc(*k1*sizeof(*p1));
   if ( p1 == NULL ) {
      printf("srht.c: memory error.\n"); exit(-1);
   }

   for (i=0; i<*k1; ++i) {
      (*p1)[i] = p[i]; // note that *p1[i] == *(p1[i]), as all postfix operators 
                       // take precedence over all prefix operators
      //printf("split: *p1[%d] = %d\n", i, (*p1)[i]);
   }

   //*k2 = len_p-*k1;
   *p2 = malloc(*k2*sizeof(*p2));
   if ( p2 == NULL ) {
      printf("srht.c: memory error.\n"); exit(-1);
   }

   for (i=0; i<*k2; ++i) {
      (*p2)[i] = p[i+*k1];
      //printf("split: *p2[%d] = %d\n", i, (*p2)[i]);
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
