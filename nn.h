#include <iostream>
#include <armadillo>
#include <vector>

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

void vec_disp(vector<mat> a)
{
	for(int i = 0; i<a.size();i++)
	{
		a[i].print(" ");
	}
}

vector<mat> init_weights(mat nodes)
{
	vector<mat> unrolled_weights;
	mat temp;
	for(int i=0;i<nodes.n_cols-1;i++)
	{
		temp=randu<mat>(nodes(i+1),nodes(i)+1);
		unrolled_weights.push_back(temp);		
	}
	return unrolled_weights;	
}

vector<mat> forward_prop(mat input, vector<mat> weights, mat nodes)
{
	mat one;
	vector<mat> a;
	one<<1;
	int n_layers = nodes.n_cols;
	input = join_horiz(one,input);
	a.push_back(input);	
	for(int i=0;i<n_layers-1;i++)
	{
		weights[i].print("weights");
		input = sigmoid(input * trans(weights[i]));
		if(i!=n_layers-2)
			input = join_horiz(one,input);
		a.push_back(input); 
	}
	return a;
}

vector<mat> back_prop(mat input, mat y, vector<mat> weights, mat nodes)
{
	vector<mat> a,d,dw,one;
	mat temp;
	a = forward_prop(input,weights,nodes);
	//~ vec_disp(a);
	int m,n,ind;
	for(int i=0;i<a.size();i++)
	{
		m = a[i].n_rows;
		n = a[i].n_cols;
		temp.ones(m,n);
		one.push_back(temp);
	}
	int length = nodes.n_cols;	
	temp = (y - a[length-1]) % a[length-1] % (one[length-1] - a[length-1]); 
	//d.insert(d.begin(),temp);
	d.push_back(temp);
	for(int i=length-3; i>=0; i--)
	{		
		ind = d.size()-1;
		temp = (d[ind] * weights[i+1]) % a[i+1] % (one[i+1] - a[i+1]);		
		d.push_back(temp);
	}
	//~ cout<<"D"<<endl;
	//~ vec_disp(d);
	
	for(int i=0;i<length-1;i++)
	{
		ind = d.size() - 1;
		if(i!=0)
		{
			int n = d[i].n_cols;
			d[i] = d[i].cols(1, n-1);
		}
	}
	for(int i=0;i<length-1;i++)
	{
	
		temp = d[ind-i].t() * a[i];  
		dw.push_back(temp);
	}
	 //~ cout<<"D"<<endl;
	//~ vec_disp(d);
	//~ cout << "A" << endl;
	//~ vec_disp(a);
	return dw;
}
