#include <iostream>
#include <armadillo>
#include <vector>
#include <fstream>
#include "nn.h"
using namespace std;
using namespace arma;

int main()
{

mat a,b;
double c;
a<<0.01<<0.04122;
b<<1<<5;
c=lms_error(a,b);
cout<<c<<endl;


}

