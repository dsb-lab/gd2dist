#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS

#include <vector>
#include <random>

//Sorting
std::vector<double> mergeSort(std::vector<double>&);

//Statistics
double mean(std::vector<double>&);
std::vector<double> percentile(std::vector<double>&, std::vector<double>&);

//Cumsum
std::vector<double> cumsum(std::vector<double>&);

//Choice pos
std::vector<int> choicepos(int, int = 1);
std::vector<int> choicepos(std::vector<double>&, int = 1);

#endif