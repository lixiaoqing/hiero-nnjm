
### Compilation options.

# C++ compiler. Tested with g++ and Intel icpc.
CXX=g++

# Architecture. Set to x86_64 or i686 to override.
ARCH:=$(shell uname -m)
# Operating system. Set to override (the only option that makes any difference is Darwin).
OS:=$(shell uname -s)

### Required libraries. You must install these prior to building.

# Set this to the root directory of Boost (should have a subdirectory named boost):
BOOST=/usr/include/boost
# Where to find Boost header files
BOOST_INC=$(BOOST)/include

### Optional libraries.

# To use the MKL library, uncomment the line below and set it to the MKL root:
#MKL=/usr/usc/intel/12.1.1/mkl
MKL=/opt/intel/mkl

# Currently, this is needed only if USE_CHRONO is defined:
# Where to find Boost libraries
BOOST_LIB=$(BOOST)/lib
# On some systems, a suffix is appended for the multithreaded version.
BOOST_LIB_SUFFIX=

BOOST_CFLAGS=-I$(BOOST_INC)
BOOST_LDFLAGS=
ifdef USE_CHRONO
  BOOST_CFLAGS+=-DUSE_CHRONO
  BOOST_LDLIBS+=-lboost_system$(BOOST_LIB_SUFFIX) -lboost_chrono$(BOOST_LIB_SUFFIX)
endif
ifdef BOOST_LDLIBS
  BOOST_LDFLAGS+=-L$(BOOST_LIB) -Wl,-rpath -Wl,$(BOOST_LIB)
endif

ifdef MKL
  MKL_CFLAGS=-I$(MKL)/include -DEIGEN_USE_MKL_ALL
  MKL_LDLIBS=-Wl,--start-group
  ifeq ($(ARCH),x86_64)
    MKL_LDFLAGS=-L$(MKL)/lib/intel64 -Wl,-rpath -Wl,$(MKL)/lib/intel64
    MKL_LDLIBS+=-lmkl_intel_lp64
  endif
  ifeq ($(ARCH),i686)
    MKL_LDFLAGS=-L$(MKL)/lib/ia32 -Wl,-rpath -Wl,$(MKL)/lib/ia32
    MKL_LDLIBS+=-lmkl_intel
  endif

  ifneq (,$(findstring g++,$(CXX)))
    MKL_LDLIBS+=-lmkl_gnu_thread
  endif
  ifneq (,$(findstring icpc,$(CXX)))
    MKL_LDLIBS+=-lmkl_intel_thread
  endif

  #MKL_LDLIBS=-lmkl_rt
  MKL_LDLIBS+=-lmkl_core -Wl,--end-group
endif

ALL_LDFLAGS= $(MKL_LDFLAGS) $(BOOST_LDFLAGS) 
ALL_LDLIBS=$(MKL_LDLIBS) $(BOOST_LDLIBS)



CXX=g++
CXXFLAGS=-std=c++0x -O3 -fopenmp -lz -DEIGEN_NO_DEBUG -I. -Ieigen -Inplm -DKENLM_MAX_ORDER=5 $(MKL_CFLAGS)
#CXXFLAGS=-std=c++0x -g -fopenmp -lz -DEIGEN_NO_DEBUG -I. -Ieigen -Inplm -DKENLM_MAX_ORDER=5 $(MKL_CFLAGS)
objs=lm/*.o util/*.o util/double-conversion/*.o

all: translator ruletable2bin
#all: translator
translator: main.o translator.o lm.o ruletable.o vocab.o cand.o myutils.o neuralLM.a $(objs)
	$(CXX) -o hiero main.o translator.o lm.o ruletable.o vocab.o myutils.o cand.o neuralLM.a $(objs) $(CXXFLAGS) $(ALL_LDFLAGS) $(ALL_LDLIBS)
ruletable2bin: ruletable2bin.o myutils.o
	$(CXX) -o ruletable2bin ruletable2bin.o myutils.o $(CXXFLAGS)

main.o: translator.h stdafx.h cand.h vocab.h ruletable.h lm.h myutils.h
translator.o: translator.h stdafx.h cand.h vocab.h ruletable.h lm.h myutils.h
lm.o: lm.h stdafx.h
ruletable.o: ruletable.h stdafx.h cand.h
vocab.o: vocab.h stdafx.h
cand.o: cand.h stdafx.h
myutils.o: myutils.h stdafx.h
ruletable2bin.o:myutils.h stdafx.h

clean:
	rm *.o
