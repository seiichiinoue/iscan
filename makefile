CC = g++
STD = -std=c++11
LLDB = -g
BOOST = -lboost_serialization
PYTHON = -lboost_python3-py36
FMATH = -fomit-frame-pointer -fno-operator-names -msse2 -mfpmath=sse -march=native
GFLAGS = -lglog -lgflags
GSL = -lgsl -lgslcblas
INCLUDE = -I/usr/include/ -pthread
LDFLAGS = `python3-config --includes` `python3-config --ldflags`

scan:
	$(CC) -O3 $(STD) -o scan train.cpp $(BOOST) $(INCLUDE) $(FMATH) $(GFLAGS)

test:
	$(CC) -O3 $(STD) -o test train.cpp $(BOOST) $(INCLUDE) $(FMATH) $(GFLAGS)

prob:
	$(CC) -O3 $(STD) -o prob scripts/prob.cpp $(BOOST) $(INCLUDE) $(FMATH) $(GFLAGS)

infer:
	$(CC) -O3 $(STD) -o infer scripts/infer.cpp $(BOOST) $(INCLUDE) $(FMATH) $(GFLAGS)

clean:
	rm -f scan test prob infer

.PHONY: clean
