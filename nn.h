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
	// Randomly initializing weights
	// It is pushed to a vector of armadillo mat types
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
	// Forward propagation while storing the activations in a vector of mats
	// The bias term '1' is added to all the layer activations except the last one (output layer)
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
	
	// Local error term for the last layer	
	temp = (y - a[length-1]) % a[length-1] % (one[length-1] - a[length-1]); 
	//d.insert(d.begin(),temp);
	
	// The local errors are pushed back one by one. The first term in the vector corresponds to the local error of the last layer
	d.push_back(temp);
	for(int i=length-3; i>=0; i--)
	{		
		ind = d.size()-1;
		temp = (d[ind] * weights[i+1]) % a[i+1] % (one[i+1] - a[i+1]);		
		d.push_back(temp);
	}
	//~ cout<<"D"<<endl;
	//~ vec_disp(d);
	
	// The local error for the bias terms need to be removed
	
	for(int i=0;i<length-1;i++)
	{
		
		if(i!=0)
		{
			// The local error for the output layer does not contain a bias term. Hence this need not be run for i=0
			int n = d[i].n_cols;
			d[i] = d[i].cols(1, n-1);
		}
	}
	// Computing error gradients
	for(int i=0;i<length-1;i++)
	{
		//As the local errors are pushed in the reversed order, the indexing is reversed again
		ind = d.size() - 1;
		temp = d[ind-i].t() * a[i];  
		dw.push_back(temp);
	}
	 //~ cout<<"D"<<endl;
	//~ vec_disp(d);
	//~ cout << "A" << endl;
	//~ vec_disp(a);
	return dw;
}
