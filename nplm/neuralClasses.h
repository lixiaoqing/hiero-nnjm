#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include <boost/unordered_map.hpp> 
#include <Eigen/Dense>
#include "maybe_omp.h"

#include "util.h"
#include "graphClasses.h"
#include "USCMatrix.h"

// classes for various kinds of layers
#include "SoftmaxLoss.h"
#include "Activation_function.h"

//#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_DEFAULT_TO_ROW_MAJOR

namespace nplm
{

	// is this cheating?
	using Eigen::Matrix;
	using Eigen::MatrixBase;
	using Eigen::Dynamic;

	typedef boost::unordered_map<int,bool> int_map;

	class Linear_layer              //linear layer
	{
		private: 
			Matrix<double,Dynamic,Dynamic> U;        //weights
			Matrix<double,Dynamic,Dynamic> U_gradient; //gradients
			Matrix<double,Dynamic,Dynamic> U_velocity; //
			Matrix<double,Dynamic,Dynamic> U_running_gradient; //

			friend class model;

		public:
			Linear_layer() { }
			Linear_layer(int rows, int cols) { resize(rows, cols); } //number of input neuros, number of output neuros

			void resize(int rows, int cols)
			{
				U.setZero(rows, cols);
				U_gradient.setZero(rows, cols);
				U_running_gradient.setZero(rows, cols);
				U_velocity.setZero(rows, cols);
			}

			void read(std::ifstream &U_file) { readMatrix(U_file, U); }  //read weights from file
			void write(std::ofstream &U_file) { writeMatrix(U, U_file); } //write weights to file

			template <typename Engine>
				void initialize(Engine &engine, bool init_normal, double init_range)
				{
					initMatrix(engine, U, init_normal, init_range); //randomly initialize the weights with normal or uniform distribution
				}	  

			int n_inputs () const { return U.cols(); }  //output neuron --> input neuron, columns denote input neuron
			int n_outputs () const { return U.rows(); } //rows denote output neuron

			template <typename DerivedIn, typename DerivedOut>
				void fProp(const MatrixBase<DerivedIn> &input, const MatrixBase<DerivedOut> &output) const //weight matrix * input
				{
					UNCONST(DerivedOut, output, my_output);
					my_output.leftCols(input.cols()).noalias() = U*input;
				}

			// Sparse input
			template <typename ScalarIn, typename DerivedOut>
				void fProp(const USCMatrix<ScalarIn> &input, const MatrixBase<DerivedOut> &output_const) const
				{
					UNCONST(DerivedOut, output_const, output);
					output.setZero();
					uscgemm(1.0, U, input, output.leftCols(input.cols())); //U*input=output
				}

			template <typename DerivedGOut, typename DerivedGIn>
				void bProp(const MatrixBase<DerivedGOut> &input, MatrixBase<DerivedGIn> &output) const
				{
					UNCONST(DerivedGIn, output, my_output);
					my_output.noalias() = U.transpose()*input; //U.transpose --> input neuron as rows, output neuron as columns
				}

			template <typename DerivedGOut, typename DerivedIn>
				void computeGradient(const MatrixBase<DerivedGOut> &bProp_input, 
						const MatrixBase<DerivedIn> &fProp_input, 
						double learning_rate, double momentum, double L2_reg)
				{
					U_gradient.noalias() = bProp_input*fProp_input.transpose(); //derivative of each output neuron * each input neuron 

					// This used to be multithreaded, but there was no measureable difference
					if (L2_reg > 0.0)
					{
						U_gradient *= 1 - 2*L2_reg;
					}
					if (momentum > 0.0) //?
					{
						U_velocity = momentum*U_velocity + U_gradient;
						U += learning_rate * U_velocity;
					}
					else
					{
						U += learning_rate * U_gradient; //update the weights
					}
				}

			template <typename DerivedGOut, typename DerivedIn>
				void computeGradientAdagrad(const MatrixBase<DerivedGOut> &bProp_input, 
						const MatrixBase<DerivedIn> &fProp_input, 
						double learning_rate, double momentum, double L2_reg) //adagradient algorithm
				{
					U_gradient.noalias() = bProp_input*fProp_input.transpose();

					if (L2_reg != 0)
					{
						U_gradient *= 1 - 2*L2_reg;
					}

					// ignore momentum?

					U_running_gradient.array() += U_gradient.array().square(); //plus all the previous gradients
					U.array() += learning_rate * U_gradient.array() / U_running_gradient.array().sqrt(); //weights update
				}

			template <typename DerivedGOut, typename DerivedIn, typename DerivedGW>
				void computeGradientCheck(const MatrixBase<DerivedGOut> &bProp_input, 
						const MatrixBase<DerivedIn> &fProp_input, 
						const MatrixBase<DerivedGW> &gradient) const
				{
					UNCONST(DerivedGW, gradient, my_gradient);
					my_gradient.noalias() = bProp_input*fProp_input.transpose();
				}
	};

	class Output_word_embeddings
	{
		private:
			// row-major is better for uscgemm
			//Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> W;
			// Having W be a pointer to a matrix allows ease of sharing
			// input and output word embeddings
			Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> *W; //weight matrix, vocab_size*output_embedding_dimension
			std::vector<double> W_data;
			Matrix<double,Dynamic,1> b; //bias
			Matrix<double,Dynamic,Dynamic> W_running_gradient; //used for adagrad
			Matrix<double,Dynamic,Dynamic> W_gradient; //weight gradient matrix
			Matrix<double,Dynamic,1> b_running_gradient; //used for adagrad
			Matrix<double,Dynamic,1> b_gradient; //bias gradient

		public:
			Output_word_embeddings() { }
			Output_word_embeddings(int rows, int cols) { resize(rows, cols); }

			void resize(int rows, int cols) //input neurons as columns, output neurons as rows, output neuron number is vocab size, input neuron number is output embedding dimension
			{
				W->setZero(rows, cols);
				b.setZero(rows);
			}
			void set_W(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> *input_W) {
				W = input_W;
			}
			void read_weights(std::ifstream &W_file) { readMatrix(W_file, *W); } //read weights from file
			void write_weights(std::ofstream &W_file) { writeMatrix(*W, W_file); } //write weights to file
			void read_biases(std::ifstream &b_file) { readMatrix(b_file, b); } //read bias from file 
			void write_biases(std::ofstream &b_file) { writeMatrix(b, b_file); } //write bias to file

			template <typename Engine>
				void initialize(Engine &engine, bool init_normal, double init_range, double init_bias)
				{
					initMatrix(engine, *W, init_normal, init_range); //initialize the weight matrix with normal or uniform distribution
					b.fill(init_bias);
				}

			int n_inputs () const { return W->cols(); }
			int n_outputs () const { return W->rows(); }

			template <typename DerivedIn, typename DerivedOut>
				void fProp(const MatrixBase<DerivedIn> &input,
						const MatrixBase<DerivedOut> &output) const
				{
					UNCONST(DerivedOut, output, my_output);
					my_output = ((*W) * input).colwise() + b; //weight matrix * input + bias
				}

			// Sparse output version
			template <typename DerivedIn, typename DerivedOutI, typename DerivedOutV>
				void fProp(const MatrixBase<DerivedIn> &input,
						const MatrixBase<DerivedOutI> &samples,
						const MatrixBase<DerivedOutV> &output) const
				{
					UNCONST(DerivedOutV, output, my_output);
					#pragma omp parallel for
					for (int instance_id = 0; instance_id < samples.cols(); instance_id++)
						for (int sample_id = 0; sample_id < samples.rows(); sample_id++)
							my_output(sample_id, instance_id) = b(samples(sample_id, instance_id));
					USCMatrix<double> sparse_output(W->rows(), samples, my_output);
					uscgemm_masked(1.0, *W, input, sparse_output);
					my_output = sparse_output.values; // too bad, so much copying
				}

			// Return single element of output matrix
			template <typename DerivedIn>
				double fProp(const MatrixBase<DerivedIn> &input, 
						int word,
						int instance) const 
				{
					return W->row(word).dot(input.col(instance)) + b(word); //W*input+b
				}

			// Dense versions (for log-likelihood loss)

			template <typename DerivedGOut, typename DerivedGIn>
				void bProp(const MatrixBase<DerivedGOut> &input_bProp_matrix,
						const MatrixBase<DerivedGIn> &bProp_matrix) const
				{
					// W is vocab_size x output_embedding_dimension
					// input_bProp_matrix is vocab_size x minibatch_size
					// bProp_matrix is output_embedding_dimension x minibatch_size
					UNCONST(DerivedGIn, bProp_matrix, my_bProp_matrix);
					my_bProp_matrix.leftCols(input_bProp_matrix.cols()).noalias() =
						W->transpose() * input_bProp_matrix;
				}

			template <typename DerivedIn, typename DerivedGOut>
				void computeGradient(const MatrixBase<DerivedIn> &predicted_embeddings,
						const MatrixBase<DerivedGOut> &bProp_input,
						double learning_rate,
						double momentum) //not sure if we want to use momentum here
				{
					// W is vocab_size x output_embedding_dimension
					// b is vocab_size x 1
					// predicted_embeddings is output_embedding_dimension x minibatch_size
					// bProp_input is vocab_size x minibatch_size

					W->noalias() += learning_rate * bProp_input * predicted_embeddings.transpose();
					b += learning_rate * bProp_input.rowwise().sum();
				}

			// Sparse versions

			template <typename DerivedGOutI, typename DerivedGOutV, typename DerivedGIn>
				void bProp(const MatrixBase<DerivedGOutI> &samples,
						const MatrixBase<DerivedGOutV> &weights,
						const MatrixBase<DerivedGIn> &bProp_matrix) const
				{
					UNCONST(DerivedGIn, bProp_matrix, my_bProp_matrix);
					my_bProp_matrix.setZero();
					uscgemm(1.0,
							W->transpose(), 
							USCMatrix<double>(W->rows(), samples, weights),
							my_bProp_matrix.leftCols(samples.cols())); // narrow bProp_matrix for possible short minibatch
				}

			template <typename DerivedIn, typename DerivedGOutI, typename DerivedGOutV>
				void computeGradient(const MatrixBase<DerivedIn> &predicted_embeddings,
						const MatrixBase<DerivedGOutI> &samples,
						const MatrixBase<DerivedGOutV> &weights,
						double learning_rate, double momentum) //not sure if we want to use momentum here
				{
					USCMatrix<double> gradient_output(W->rows(), samples, weights);
					uscgemm(learning_rate,
							gradient_output,
							predicted_embeddings.leftCols(gradient_output.cols()).transpose(),
							*W); // narrow predicted_embeddings for possible short minibatch
					uscgemv(learning_rate,
							gradient_output,
							Matrix<double,Dynamic,1>::Ones(gradient_output.cols()),
							b);
				}

			template <typename DerivedIn, typename DerivedGOutI, typename DerivedGOutV>
				void computeGradientAdagrad(const MatrixBase<DerivedIn> &predicted_embeddings,
						const MatrixBase<DerivedGOutI> &samples,
						const MatrixBase<DerivedGOutV> &weights,
						double learning_rate, double momentum) //not sure if we want to use momentum here
				{
					W_gradient.setZero(W->rows(), W->cols());
					b_gradient.setZero(b.size());
					if (W_running_gradient.rows() != W->rows() || W_running_gradient.cols() != W->cols())
						W_running_gradient.setZero(W->rows(), W->cols());
					if (b_running_gradient.size() != b.size())
						b_running_gradient.setZero(b.size());

					USCMatrix<double> gradient_output(W->rows(), samples, weights);
					uscgemm(learning_rate,
							gradient_output,
							predicted_embeddings.leftCols(samples.cols()).transpose(),
							W_gradient);
					uscgemv(learning_rate, gradient_output,
							Matrix<double,Dynamic,1>::Ones(weights.cols()),
							b_gradient);

					int_map update_map; //stores all the parameters that have been updated
					for (int sample_id=0; sample_id<samples.rows(); sample_id++)
						for (int train_id=0; train_id<samples.cols(); train_id++)
							update_map[samples(sample_id, train_id)] = 1;

					// Convert to std::vector for parallelization
					std::vector<int> update_items;
					for (int_map::iterator it = update_map.begin(); it != update_map.end(); ++it)
						update_items.push_back(it->first);
					int num_items = update_items.size();

					#pragma omp parallel for
					for (int item_id=0; item_id<num_items; item_id++)
					{
						int update_item = update_items[item_id];
						W_running_gradient.row(update_item).array() += W_gradient.row(update_item).array().square();
						b_running_gradient(update_item) += b_gradient(update_item) * b_gradient(update_item);
						W->row(update_item).array() += learning_rate * W_gradient.row(update_item).array() / W_running_gradient.row(update_item).array().sqrt();
						b(update_item) += learning_rate * b_gradient(update_item) / sqrt(b_running_gradient(update_item));
					}
				}

			template <typename DerivedIn, typename DerivedGOutI, typename DerivedGOutV, typename DerivedGW, typename DerivedGb>
				void computeGradientCheck(const MatrixBase<DerivedIn> &predicted_embeddings,
						const MatrixBase<DerivedGOutI> &samples,
						const MatrixBase<DerivedGOutV> &weights,
						const MatrixBase<DerivedGW> &gradient_W,
						const MatrixBase<DerivedGb> &gradient_b) const
				{
					UNCONST(DerivedGW, gradient_W, my_gradient_W);
					UNCONST(DerivedGb, gradient_b, my_gradient_b);
					my_gradient_W.setZero();
					my_gradient_b.setZero();
					USCMatrix<double> gradient_output(W->rows(), samples, weights);
					uscgemm(1.0,
							gradient_output,
							predicted_embeddings.leftCols(samples.cols()).transpose(),
							my_gradient_W);
					uscgemv(1.0, gradient_output,
							Matrix<double,Dynamic,1>::Ones(weights.cols()), my_gradient_b);
				}
	};

	class Input_word_embeddings
	{
		private:
			Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> *W; //weight matrix: vocab_size*input_embedding_dimension
			int context_size, vocab_size;  //the used context size, the vocab_size
			Matrix<double,Dynamic,Dynamic> W_running_gradient; //weight gradient used for adagrad
			Matrix<double,Dynamic,Dynamic> W_gradient; //weight gradient

			Matrix<double, Dynamic, Dynamic, Eigen::RowMajor> *sent_embeddings;

			friend class model;

		public:
			Input_word_embeddings() : context_size(0), vocab_size(0) { }
			Input_word_embeddings(int rows, int cols, int context) { resize(rows, cols, context); }

			void set_W(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> *input_W) {
				W = input_W;
			}

			void set_sent_embeddings(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> *embeddings) {
				sent_embeddings = embeddings;
			}

			void resize(int rows, int cols, int context)
			{
				context_size = context;
				vocab_size = rows;
				W->setZero(rows, cols);
			}

			void sent_embeddings_resize(int rows, int cols)
			{
				sent_embeddings->setZero(rows, cols);
			}

			void read(std::ifstream &W_file) { readMatrix(W_file, *W); }
			void readSentEmbeddings(const std::string &sentembedfilename) 
			{
				std::ifstream sentembedfile(sentembedfilename.c_str());
				if( !sentembedfile )
				{
					std::cerr<<"Cannot open the sentence embedding file from: "<<sentembedfilename<<std::endl;
					exit(-1);
				}
				std::string firstline = "";
				getline(sentembedfile, firstline);
				std::istringstream buf(firstline);
				int sent_counts = 0, sent_embed_dim = 0;
				if( !(buf>>sent_counts>>sent_embed_dim) )
				{
					std::cerr<<"Sentence embedding format error for: "<<firstline<<std::endl;
					exit(-1);
				}
				sent_embeddings->setZero(sent_counts, sent_embed_dim);
				readMatrix(sentembedfile, *sent_embeddings);
			}	
			void write(std::ofstream &W_file) { writeMatrix(*W, W_file); }

			template <typename Engine>
				void initialize(Engine &engine, bool init_normal, double init_range)
				{
					initMatrix(engine,
							*W,
							init_normal,
							init_range); //initialize the weight matrix using normal or uniform distribution
				}

			int n_inputs() const { return -1; }
			int n_outputs() const { return W->cols() * context_size; } //concatenate the embedding for each context word

			// set output_id's embedding to the weighted average of all embeddings
			template <typename Dist>
				void average(const Dist &dist, int output_id)
				{
					W->row(output_id).setZero();
					for (int i=0; i < W->rows(); i++)
						if (i != output_id)
							W->row(output_id) += dist.prob(i) * W->row(i); //dist.prob as weights
				}

			template <typename DerivedIn, typename DerivedOut>
				void fProp(const MatrixBase<DerivedIn> &input,
						const MatrixBase<DerivedOut> &output) const
				{
					int embedding_dimension = W->cols();

					// W      is vocab_size                        x embedding_dimension
					// input  is ngram_size*vocab_size             x minibatch_size
					// output is ngram_size*embedding_dimension x minibatch_size

					/* 
					// Dense version:
					for (int ngram=0; ngram<context_size; ngram++)
					output.middleRows(ngram*embedding_dimension, embedding_dimension) = W.transpose() * input.middleRows(ngram*vocab_size, vocab_size);
					*/

					UNCONST(DerivedOut, output, my_output);
					my_output.setZero();
					//first id in ngram denotes the sent_id
					uscgemm(1.0,
							sent_embeddings->transpose(),
							USCMatrix<double>(sent_embeddings->rows(), input.middleRows(0, 1), Matrix<double,1,Dynamic>::Ones(input.cols())),
							my_output.block(0, 0, embedding_dimension, input.cols()));

					for (int ngram=1; ngram<context_size; ngram++)
					{
						// input might be narrower than expected due to a short minibatch,
						// so narrow output to match
						uscgemm(1.0,
								W->transpose(), 
								USCMatrix<double>(W->rows(),input.middleRows(ngram, 1),Matrix<double,1,Dynamic>::Ones(input.cols())),
								my_output.block(ngram*embedding_dimension, 0, embedding_dimension, input.cols())); //embedding concatenation
					}
				}

			// When model is premultiplied, this layer doesn't get used,
			// but this method is used to get the input into a sparse matrix.
			// Hopefully this can get eliminated someday
			template <typename DerivedIn, typename ScalarOut>
				void munge(const MatrixBase<DerivedIn> &input, USCMatrix<ScalarOut> &output) const
				{
					output.resize(vocab_size*context_size, context_size, input.cols());
					for (int i=0; i < context_size; i++)
						output.indexes.row(i).array() = input.row(i).array() + i*vocab_size;
					output.values.fill(1.0);
				}

			template <typename DerivedGOut, typename DerivedIn>
				void computeGradient(const MatrixBase<DerivedGOut> &bProp_input,
						const MatrixBase<DerivedIn> &input_words,
						double learning_rate, double momentum, double L2_reg)
				{
					int embedding_dimension = W->cols();

					// W           is vocab_size                        x embedding_dimension
					// input       is ngram_size*vocab_size             x minibatch_size
					// bProp_input is ngram_size*embedding_dimension x minibatch_size

					/*
					// Dense version:
					for (int ngram=0; ngram<context_size; ngram++)
					W += learning_rate * input_words.middleRows(ngram*vocab_size, vocab_size) * bProp_input.middleRows(ngram*embedding_dimension, embedding_dimension).transpose()
					*/

					//since the first id in ngram denotes the sent_id, updating the word embeddings from the second id in ngram
					for (int ngram=1; ngram<context_size; ngram++)
					{
						uscgemm(learning_rate, 
								USCMatrix<double>(W->rows(), input_words.middleRows(ngram, 1), Matrix<double,1,Dynamic>::Ones(input_words.cols())),
								bProp_input.block(ngram*embedding_dimension,0,embedding_dimension,input_words.cols()).transpose(),
								*W); //for each context word, update the gradient
					}
				}

			template <typename DerivedGOut, typename DerivedIn>
				void computeGradientAdagrad(const MatrixBase<DerivedGOut> &bProp_input,
						const MatrixBase<DerivedIn> &input_words,
						double learning_rate, double momentum, double L2_reg)
				{
					int embedding_dimension = W->cols();

					W_gradient.setZero(W->rows(), W->cols());
					if (W_running_gradient.rows() != W->rows() || W_running_gradient.cols() != W->cols())
						W_running_gradient.setZero(W->rows(), W->cols());

					//since the first id in ngram denotes the sent_id, updating the word embeddings from the second id in ngram
					for (int ngram=1; ngram<context_size; ngram++)
					{
						uscgemm(learning_rate, 
								USCMatrix<double>(W->rows(),input_words.middleRows(ngram, 1),Matrix<double,1,Dynamic>::Ones(input_words.cols())),
								bProp_input.block(ngram*embedding_dimension, 0, embedding_dimension, input_words.cols()).transpose(),
								W_gradient);
					}

					int_map update_map; //stores all the parameters that have been updated

					for (int train_id=0; train_id<input_words.cols(); train_id++)
					{
						update_map[input_words(train_id)] = 1;
					}

					// Convert to std::vector for parallelization
					std::vector<int> update_items;
					for (int_map::iterator it = update_map.begin(); it != update_map.end(); ++it)
					{
						update_items.push_back(it->first);
					}
					int num_items = update_items.size();

					#pragma omp parallel for
					for (int item_id=0; item_id<num_items; item_id++)
					{
						int update_item = update_items[item_id];
						W_running_gradient.row(update_item).array() += W_gradient.row(update_item).array().square();
						W->row(update_item).array() += learning_rate * W_gradient.row(update_item).array() / W_running_gradient.row(update_item).array().sqrt();
					}
				}

			template <typename DerivedGOut, typename DerivedIn, typename DerivedGW>
				void computeGradientCheck(const MatrixBase<DerivedGOut> &bProp_input,
						const MatrixBase<DerivedIn> &input_words,
						int x, int minibatch_size,
						const MatrixBase<DerivedGW> &gradient) const //not sure if we want to use momentum here
				{
					UNCONST(DerivedGW, gradient, my_gradient);
					int embedding_dimension = W->cols();
					my_gradient.setZero();
					//since the first id in ngram denotes the sent_id, updating the word embeddings from the second id in ngram
					for (int ngram=1; ngram<context_size; ngram++)
						uscgemm(1.0, 
								USCMatrix<double>(W->rows(),input_words.middleRows(ngram, 1),Matrix<double,1,Dynamic>::Ones(input_words.cols())),
								bProp_input.block(ngram*embedding_dimension, 0, embedding_dimension, input_words.cols()).transpose(),
								my_gradient);
				}
	};

} // namespace nplm
