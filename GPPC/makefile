all: main

main: main.o ScenarioLoader.o
	g++ -o main main.o ScenarioLoader.o

ScenarioLoader.o: ScenarioLoader.cpp ScenarioLoader.h
	g++ -c ScenarioLoader.cpp

main.o: main.cpp
	g++ -c main.cpp

clean:
	rm main.o ScenarioLoader.o main