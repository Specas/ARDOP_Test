#include <iostream>
#include <armadillo>
#include "nn.h"
using namespace std;
using namespace arma;

int main()
{
	mat A,B;
	A<<0<<0<<0;
	B = sigmoid(A);
	B.print();
	return 0;
}