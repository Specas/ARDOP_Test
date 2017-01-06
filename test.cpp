#include <iostream>
#include <armadillo>
#include <vector>
#include "nn.h"
using namespace std;
using namespace arma;

int main()
{
	// Initializing inputs, nodes and outputs
	
	mat in,nodes,y; 
	vector<mat> out;
	in<<1<<3;
	nodes<<2<<3<<2;
	y<<1<<0;
	vector<mat> w;
	w = init_weights(nodes);
	out = back_prop(in,y,w,nodes);
	for(int i = 0; i<out.size();i++)
	{
		out[i].print("a");
	}

}

