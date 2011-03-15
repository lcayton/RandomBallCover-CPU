CC=gcc
#other flags -O3 -funroll-loops  -msse -g
CCFLAGS= -fopenmp -funroll-loops -msse -Wall -O3
SOURCES= scaleDriver.c utils.c brute.c rbc.c #hashDriver.c driver.c compare_build.c exactDriver.c scaleDriver.c 
LINKFLAGS= -lgsl -lgslcblas -lm
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=testCRBC
all: $(SOURCES) $(CUSOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CCFLAGS) $(OBJECTS) -o $@ $(LINKFLAGS)

%.o:%.c
	$(CC) $(CCFLAGS) -c $+ 

clean:
	rm -f *.o
	rm -f $(EXECUTABLE)
