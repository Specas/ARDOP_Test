#include <iostream>
#include <armadillo>
#include <vector>
#include <fstream>
#include "nn.h"
using namespace std;
using namespace arma;

int main()
{
	// Initializing inputs, nodes and outputs
	ifstream in_file ("input.txt");
	ifstream o_file ("output.txt");
	string line;
	mat temp,t_out;
	vector<mat> in_vec,out_vec,w,delta,output;
	mat x,nodes,y;
	if(in_file.is_open())
	{
		while(getline(in_file,line))
		{
			temp = convert_to_int(line);
			in_vec.push_back(temp);
		}
		in_file.close();
	}
	if(o_file.is_open())
	{
		while(getline(o_file,line))
		{
			temp = convert_to_int(line);
			out_vec.push_back(temp);
		}
		o_file.close();
	}
	
	float alpha = 1;
	int iters = 500;
	double error;
	nodes<<4<<10<<1;

	w = init_weights(nodes);
	for(int i = 0;i<iters;i++)
	{
		double acc=0,t_error;
		for(int j=0;j<in_vec.size()-1;j++)
		{
			x = in_vec[j];
			y = out_vec[j];
			output = forward_prop(x,w,nodes);
			t_out = threshold(output[(output.size()-1)]);
			//t_out = output[(output.size()-1)];
			error = lms_error(y,t_out);
			acc= acc + error;
			//cout<<"ERROR "<<i<<":"<<error<<endl;
			delta = back_prop(x,y,w,nodes);
			for(int i=0;i<w.size();i++)
			{
				w[i] = (alpha * delta[i]) + w[i];
			}


		}
		t_error=acc/(in_vec.size()-1);
		cout<<"T_ERROR "<<i<<":"<<t_error<<endl;
	}
	mat x_test,y_test;
	vector<mat> y_temp;
	x_test<<0<<1<<1<<1;
	y_temp = forward_prop(x_test,w,nodes);
	y_test = threshold(y_temp[(y_temp.size()-1)]);
	cout<<y_test;
}

