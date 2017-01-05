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

mat forward_prop(mat input, vector<mat> weights, mat nodes)
{
	mat one;
	one<<1;
	int n_layers = nodes.n_cols;	
	for(int i=0;i<n_layers-1;i++)
	{
		weights[i].print("weights");
		input = join_horiz(one,input);
		input = sigmoid(input * trans(weights[i])); 
		input.print();
	}
	return input;
}



// for (int i = 1; i < argc; ++i)
//     {
//         if (string(argv[i]) == "--cascade")
//             cascadeName = argv[++i];
//         else if (string(argv[i]) == "--video")
//         {
//             inputName = argv[++i];
//             isInputVideo = true;
//         }
//         else if (string(argv[i]) == "--camera")
//         {
//             inputName = argv[++i];
//             isInputCamera = true;
//         }
//         else if (string(argv[i]) == "--help")
//         {
//             help();
//             return -1;
//         }
//         else if (!isInputImage)
//         {
//             inputName = argv[i];
//             isInputImage = true;
//         }
//         else
//         {
//             cout << "Unknown key: " << argv[i] << endl;
//             return -1;
//         }
//     }