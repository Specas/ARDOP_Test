#include <iostream>
#include <armadillo>
#include <vector>
#include "nn.h"
using namespace std;
using namespace arma;

int main()
{
	mat in,out,nodes; 
	in<<1<<3;
	nodes<<2<<2<<1;
	vector<mat> w;
	w = init_weights(nodes);
	out = forward_prop(in,w,nodes);
	out.print();

	return 0;
}

