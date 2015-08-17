#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <string>
#include <boost/random/mersenne_twister.hpp>

#include "neuralClasses.h"
#include "Activation_function.h"

namespace nplm
{

	class model {  //structure of the neural network
		public:
			Input_word_embeddings input_layer;
			Linear_layer first_hidden_linear;
			Activation_function first_hidden_activation;
			Linear_layer second_hidden_linear;
			Activation_function second_hidden_activation;
			Output_word_embeddings output_layer;
			Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> output_embedding_matrix,
				input_embedding_matrix,
				input_and_output_embedding_matrix;

			activation_function_type activation_function;
			int ngram_size, input_vocab_size, output_vocab_size, input_embedding_dimension, num_hidden, output_embedding_dimension;
			bool premultiplied;

			model(int ngram_size,
					int input_vocab_size,
					int output_vocab_size,
					int input_embedding_dimension,
					Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> *sent_embeddings,
					int num_hidden,
					int output_embedding_dimension,
					bool share_embeddings) //model construction
			{
				if (share_embeddings){ //input_embedding_matrix and output_embedding_matrix are the same, using only one matrix
					input_and_output_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
					input_layer.set_W(&input_and_output_embedding_matrix); //weight matrix of the input layer is the input_embedding_matrix
					input_layer.set_sent_embeddings(sent_embeddings);
					output_layer.set_W(&input_and_output_embedding_matrix);//weight matrix of the output layer is the output_embedding_matrix
				}
				else { //input_embedding_matrix is different from the output_embedding_matrix
					input_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
					output_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
					input_layer.set_W(&input_embedding_matrix);
					input_layer.set_sent_embeddings(sent_embeddings);
					output_layer.set_W(&output_embedding_matrix);
				}
				resize(ngram_size,
						input_vocab_size,
						output_vocab_size,
						input_embedding_dimension,
						num_hidden,
						output_embedding_dimension);
			}
			model() : ngram_size(1), 
			premultiplied(false),
			activation_function(Rectifier),
			output_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>()),
			input_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>()) //default model construction
		{
			output_layer.set_W(&output_embedding_matrix);
			input_layer.set_W(&input_embedding_matrix);
		}

			void resize(int ngram_size,
					int input_vocab_size,
					int output_vocab_size,
					int input_embedding_dimension,
					int num_hidden,
					int output_embedding_dimension); //reconstruct the network

			void initialize(boost::random::mt19937 &init_engine,
					bool init_normal,
					double init_range,
					double init_bias); //initialize the network
			void set_activation_function(activation_function_type f) //choose the activation function
			{
				activation_function = f;
				first_hidden_activation.set_activation_function(f);
				second_hidden_activation.set_activation_function(f);
			}

			void premultiply(); //whether pre-compute the first hidden linear layer

			// Since the vocabulary is not essential to the model,
			// we need a version with and without a vocabulary.
			// If the number of "extra" data structures like this grows,
			// a better solution is needed

			void read(const std::string &filename); //read the network form a file
			void read(const std::string &filename, const std::string &sentembed_filename); //read the network form a file
			void read(const std::string &filename, std::vector<std::string> &input_words, std::vector<std::string> &output_words); //read network, input words and output words from a file
			void write(const std::string &filename, const std::vector<std::string> &input_words, const std::vector<std::string> &output_words); //write the network, input words and output words to a file
			void write(const std::string &filename);

		private:
			void readConfig(std::ifstream &config_file);
			void readConfig(const std::string &filename); //read the config file
			void write(const std::string &filename, const std::vector<std::string> *input_pwords, const std::vector<std::string> *output_pwords); //write the network, input words and output words to a file
	};

} //namespace nplm

#endif
