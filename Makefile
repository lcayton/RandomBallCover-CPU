CC=gcc
#other flags -O3 -funroll-loops  -msse -g
CCFLAGS= -fopenmp -O3 -funroll-loops -msse -Wall
SOURCES= compare_build.c utils.c brute.c rbc.c #hashDriver.c exactDriver.c driver.c
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
