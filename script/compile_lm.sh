#!/bin/bash

CXX=g++
CXXFLAGS="-std=c++0x -O3 -fopenmp -lz -I. -DKENLM_MAX_ORDER=5"

#Grab all cc files in these directories except those ending in test.cc or main.cc
for i in util/double-conversion/*.cc util/*.cc lm/*.cc; do
  if [ "${i%test.cc}" == "$i" ] && [ "${i%main.cc}" == "$i" ]; then
    $CXX $CXXFLAGS -c $i -o ${i%.cc}.o
  fi
done
