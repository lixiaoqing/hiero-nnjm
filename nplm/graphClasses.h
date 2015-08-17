//creating the structure of the nn in a graph that will help in performing backpropagation and forward propagation
#pragma once

#include <cstdlib>
#include "neuralClasses.h"
#include <Eigen/Dense>

namespace nplm
{

	template <class X>
		class Node {
			public:
				X * param; //different type of layer: input_embedding_layer, linear_layer, activation_layer, output_embedding_layer
				//vector <void *> children;
				//vector <void *> parents;
				Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> fProp_matrix; //used for forward propagation
				Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> bProp_matrix; //used for backward propagation
				int minibatch_size;

			public:
				Node() : param(NULL), minibatch_size(0) { }

				Node(X *input_param, int minibatch_size)
					: param(input_param),
					minibatch_size(minibatch_size)
			{
				resize(minibatch_size);
			}

				void resize(int minibatch_size)
				{
					this->minibatch_size = minibatch_size;
					if (param->n_outputs() != -1)
					{
						fProp_matrix.setZero(param->n_outputs(), minibatch_size); //initialize the parameters for forward parameters
					}
					if (param->n_inputs() != -1)
					{
						bProp_matrix.setZero(param->n_inputs(), minibatch_size); //initialize the parameters for backward parameters
					}
				}

				void resize() { resize(minibatch_size); }

				/*
				   void Fprop(Matrix<double,Dynamic,Dynamic> & input,int n_cols)
				   {
				   param->fProp(input,fProp_matrix,0,0,n_cols);
				   }
				   void Fprop(Matrix<double,1,Dynamic> & input,int n_cols)
				   {
				   param->fProp(input,fProp_matrix,0,0,n_cols);
				   }
				   */
				//for f prop, just call the fProp node of the particular parameter. 

		};

} // namespace nplm
