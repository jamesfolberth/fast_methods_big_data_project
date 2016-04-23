
CFLAGS=-Wall -fPIC -O2
LDFLAGS= 
LDLIBS= -lc

# These are for James' laptop
#MKLROOT=/opt/intel/mkl
#CFLAGS=-Wall -fPIC -O2 -m64 -I${MKLROOT}/include
#LDFLAGS= 
#LDLIBS= -lc -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl

.PHONY: all clean

all: fht.so

fht.o: fht.c
	gcc $(CFLAGS) $(LDLIBS) $(LDFLAGS) -c $<

fht.so: fht.o
	ld -shared $< -o $@ -fPIC $(LDLIBS) $(LDFLAGS)

clean:
	rm -f *.o
	rm -f fht.so
