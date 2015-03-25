CXX=g++
CXXFLAGS=-std=c++0x -O3 -fopenmp -lz -I. -DKENLM_MAX_ORDER=6
objs=lm/*.o util/*.o util/double-conversion/*.o

all: translator phrase2bin
translator: main.o translator.o lm.o ruletable.o vocab.o cand.o myutils.o $(objs)
	$(CXX) -o hiero main.o translator.o lm.o ruletable.o vocab.o myutils.o cand.o $(objs) $(CXXFLAGS)
phrase2bin: phrase2bin.o myutils.o
	$(CXX) -o phrase2bin phrase2bin.o myutils.o $(CXXFLAGS)

main.o: translator.h stdafx.h cand.h vocab.h ruletable.h lm.h myutils.h
translator.o: translator.h stdafx.h cand.h vocab.h ruletable.h lm.h myutils.h
lm.o: lm.h stdafx.h
ruletable.o: ruletable.h stdafx.h cand.h
vocab.o: vocab.h stdafx.h
cand.o: cand.h stdafx.h
myutils.o: myutils.h stdafx.h
phrase2bin.o:myutils.h stdafx.h

clean:
	rm *.o