#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS

#include <vector>
#include <random>

//Cumsum
std::vector<double> cumsum(std::vector<double>&);

//Choice pos
std::vector<int> choicepos(int, int = 1);
std::vector<int> choicepos(std::vector<double>&, int = 1);

#endif