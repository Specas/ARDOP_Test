#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

mat sigmoid(mat in)
{
	int m = in.n_rows;
	int n = in.n_cols;
	mat out;
	out.ones(m,n);
	out = out + exp((-1*in));
	out = 1/out;
	return out;
}