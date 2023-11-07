CC = g++
STD = -std=c++11
LLDB = -g
BOOST = -lboost_serialization
PYTHON = -lboost_python3-py36
GFLAGS = -lglog -lgflags
GSL = -lgsl -lgslcblas
INCLUDE = -I/usr/include/ -pthread
LDFLAGS = `python3-config --includes` `python3-config --ldflags`
PGINCLUDE = -I./src/pg/ -I./src/pg/include/
PGLIB = -lm src/pg/InvertY.cpp src/pg/PolyaGamma.cpp src/pg/PolyaGammaAlt.cpp src/pg/PolyaGammaSP.cpp src/pg/PolyaGammaSmallB.cpp src/pg/include/RNG.cpp src/pg/include/GRNG.cpp

scan:
	$(CC) -O3 $(STD) -o scan train.cpp $(BOOST) $(INCLUDE) $(PGINCLUDE) $(PGLIB) $(GSL) $(GFLAGS) 

test:
	$(CC) -O3 $(STD) -o test train.cpp $(BOOST) $(INCLUDE) $(PGINCLUDE) $(PGLIB) $(GSL) $(GFLAGS) 

prob:
	$(CC) -O3 $(STD) -o prob tools/prob.cpp $(BOOST) $(INCLUDE) $(PGINCLUDE) $(PGLIB) $(GSL) $(GFLAGS) 

eval:
	$(CC) -O3 $(STD) -o eval tools/eval.cpp $(BOOST) $(INCLUDE) $(PGINCLUDE) $(PGLIB) $(GSL) $(GFLAGS) 

infer:
	$(CC) -O3 $(STD) -o infer tools/infer.cpp $(BOOST) $(INCLUDE) $(PGINCLUDE) $(PGLIB) $(GSL) $(GFLAGS) 

clean:
	rm -f scan test prob infer

.PHONY: clean
