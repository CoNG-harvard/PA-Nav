#include "ScenarioLoader.h"
#include <iostream>
using namespace std; 

int main(int argc, char const *argv[])
{
	/* code */
	ScenarioLoader loader("arena.map.scen");
	
	cout<<"Map loaded."<<endl;
	
	cout<<"Scenario Name:"<<loader.GetScenarioName()<<endl;

	cout<<"Number of experiments:"<<loader.GetNumExperiments()<<endl;
	

	return 0;
}