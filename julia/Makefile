
# Vanilla
#CFLAGS=-Wall -fPIC -O2 
#LDFLAGS= 
#LDLIBS= -lc

# MKL
# These are for James' laptop
#MKLROOT=/opt/intel/mkl
#CFLAGS=-Wall -fPIC -O2 -m64 -I${MKLROOT}/include
#LDFLAGS= 
#LDLIBS= -lc -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl

## SIMD
## These are for James' laptop
CFLAGS=-Wall -fPIC -O2 -msse2 -march=native
LDFLAGS= 
LDLIBS= -lc -lm


.PHONY: all clean

all: fht.so srht.so

main: srht.c 
	gcc $(CFLAGS) $(LDLIBS) $(LDFLAGS) -g -o main srht.c

fht.o: fht.c
	gcc $(CFLAGS) $(LDLIBS) $(LDFLAGS) -c $<

fht.so: fht.o
	ld -shared $< -o $@ -fPIC $(LDLIBS) $(LDFLAGS)

srht.o: srht.c
	gcc $(CFLAGS) $(LDLIBS) $(LDFLAGS) -c $<

srht.so: srht.o
	ld -shared $< -o $@ -fPIC $(LDLIBS) $(LDFLAGS)


clean:
	rm -f *.o
	rm -f fht.so srht.so main
