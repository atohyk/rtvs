rtvsmake: main.cpp
	g++ main.cpp $(shell pkg-config --libs opencv)