CXX = clang++
CFLAGS = -O2 -g -fPIC -std=c++11
INCL = -Iinclude `python-config --includes`

akima: akima.o
	$(CXX) $(INCL) $(CFLAGS) -dynamiclib -undefined dynamic_lookup -shared akima.o -o _akima.so

.cpp.o:
	$(CXX) $(INCL) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o _akima.so
