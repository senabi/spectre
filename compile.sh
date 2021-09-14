#!/bin/sh

clang++ -O3 MeasureReadLatency_x86_64.S spectre_safeside.cc -o spectre_pht
#cc -lstdc++ MeasureReadLatency_x86_64.S spectre_safeside.cc -o spectre_pht
