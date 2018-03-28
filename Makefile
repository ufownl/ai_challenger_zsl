CXX				= g++
LD				= g++
CXXFLAGS	= -std=c++11 -Wall -O3 -mavx -ftemplate-depth=1024
LDFLAGS		= -ldlib -lblas -llapack -pthread
SOURCES		= $(wildcard *.cpp)
TARGETS		= $(SOURCES:.cpp=)

all: $(TARGETS)

%: %.o
	$(LD) $^ $(LDFLAGS) -o $@

-include $(SOURCES:.cpp=.d)

%.d: %.cpp
	$(CXX) -M $(CXXFLAGS) $^ > $@

clean:
	$(RM) *.o
	$(RM) *.d
	$(RM) $(TARGETS)

.PHONY: all clean
