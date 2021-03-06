#ifndef NEURALLM_H
#define NEURALLM_H

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <boost/lexical_cast.hpp>

#include <Eigen/Dense>

#include "param.h"
#include "util.h"
#include "model.h"
#include "propagator.h"
#include "neuralClasses.h"
#include "vocabulary.h"

namespace nplm
{

	class neuralLM 
	{
		bool normalization;  //whether need normalization
		char map_digits; //map all the digit string into the same token

		vocabulary input_vocab, output_vocab; //input and output vocabulary
		model nn; //neural network model, input_layer+hidden_layer+hidden_layer+output_layer
		propagator prop; //used for propagation algorithm: forward and backward

		int ngram_size; //ngram order
		int width;  //minibatch size

		double weight; //if not log_e

		private:
		std::size_t cache_size;
		Eigen::Matrix<int,Dynamic,Dynamic> cache_keys;
		std::vector<double> cache_values;
		int cache_lookups, cache_hits;

		Eigen::Matrix<int,Eigen::Dynamic,1> ngram; // buffer for lookup_ngram
		int start, null;

		public:
		neuralLM() 
			: ngram_size(1), 
			normalization(false),
			weight(1.),
			map_digits(0),
			width(1),
			prop(nn, 1),
			cache_size(0)
		{ 
		}

		void set_normalization(bool value) { normalization = value; }
		void set_log_base(double value) { weight = 1./std::log(value); } //if not loge
		void set_map_digits(char value) { map_digits = value; }

		// This must be called if the underlying model is resized.
		void resize() {
			ngram_size = nn.ngram_size;
			ngram.setZero(ngram_size);
			if (cache_size)
			{
				cache_keys.resize(ngram_size, cache_size);
				cache_keys.fill(-1);
			}
			prop.resize();
		}

		void set_width(int width)
		{
			this->width = width;
			prop.resize(width);
		}

		void set_input_vocabulary(const vocabulary &vocab) //input vocabulary
		{
			this->input_vocab = vocab;
			start = input_vocab.lookup_word("<s>");
			null = input_vocab.lookup_word("<null>");
		}

		void set_output_vocabulary(const vocabulary &vocab) //output vocabulary
		{
			this->output_vocab = vocab;
		}

		const vocabulary &get_vocabulary() const { return this->input_vocab; } //get input vocabulary

		int lookup_input_word(const std::string &word) const //lookup word in the input vocabulary
		{
			if (map_digits) //whether map digit words into the same token
				for (int i=0; i<word.length(); i++)
					if (isdigit(word[i]))
					{
						std::string mapped_word(word);
						for (; i<word.length(); i++)
							if (isdigit(word[i]))
								mapped_word[i] = map_digits;
						return input_vocab.lookup_word(mapped_word);
					}
			return input_vocab.lookup_word(word);
		}

		int lookup_word(const std::string &word) const
		{
			return lookup_input_word(word);
		}

		int lookup_output_word(const std::string &word) const //lookup word in the output vocabulary
		{
			if (map_digits)
				for (int i=0; i<word.length(); i++)
					if (isdigit(word[i]))
					{
						std::string mapped_word(word);
						for (; i<word.length(); i++)
							if (isdigit(word[i]))
								mapped_word[i] = map_digits;
						return output_vocab.lookup_word(mapped_word);
					}
			return output_vocab.lookup_word(word);
		}

		template <typename Derived>
			double lookup_ngram(const Eigen::MatrixBase<Derived> &ngram)
			{
				assert (ngram.rows() == ngram_size);
				assert (ngram.cols() == 1);

				std::size_t hash;
				if (cache_size)
				{
					// First look in cache
					hash = Eigen::hash_value(ngram) % cache_size; // defined in util.h
					cache_lookups++;
					if (cache_keys.col(hash) == ngram)
					{
						cache_hits++;
						return cache_values[hash];
					}
				}

				// Make sure that we're single threaded. Multithreading doesn't help,
				// and in some cases can hurt quite a lot
				int save_threads = omp_get_max_threads();
				omp_set_num_threads(1);
				int save_eigen_threads = Eigen::nbThreads();
				Eigen::setNbThreads(1);
				#ifdef __INTEL_MKL__
				int save_mkl_threads = mkl_get_max_threads();
				mkl_set_num_threads(1);
				#endif

				prop.fProp(ngram.col(0)); //ngram-1 context to embeddings and then propagate

				int output = ngram(ngram_size-1, 0);
				double log_prob;

				start_timer(3);
				if (normalization) //whether do the normalization
				{
					Eigen::Matrix<double,Eigen::Dynamic,1> scores(output_vocab.size());
					prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);
					double logz = logsum(scores.col(0));
					log_prob = weight * (scores(output, 0) - logz);
				}
				else //only get the score of the current word
				{
					log_prob = weight * prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, output, 0);
				}
				stop_timer(3);

				if (cache_size)
				{
					// Update cache
					cache_keys.col(hash) = ngram;
					cache_values[hash] = log_prob;
				}

				#ifdef __INTEL_MKL__
				mkl_set_num_threads(save_mkl_threads);
				#endif
				Eigen::setNbThreads(save_eigen_threads);
				omp_set_num_threads(save_threads);

				return log_prob;
			}

		// Look up many n-grams in parallel.
		template <typename DerivedA, typename DerivedB>
			void lookup_ngram(const Eigen::MatrixBase<DerivedA> &ngram, const Eigen::MatrixBase<DerivedB> &log_probs_const)
			{
				UNCONST(DerivedB, log_probs_const, log_probs);
				assert (ngram.rows() == ngram_size);
				assert (ngram.cols() <= width); //minibatch

				prop.fProp(ngram);

				if (normalization) //softmax
				{
					Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> scores(output_vocab.size(), ngram.cols());
					prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);

					// And softmax and loss
					Matrix<double,Dynamic,Dynamic> output_probs(nn.output_vocab_size, ngram.cols());
					double minibatch_log_likelihood;
					SoftmaxLogLoss().fProp(scores.leftCols(ngram.cols()), ngram.row(nn.ngram_size-1), output_probs, minibatch_log_likelihood);
					for (int j=0; j<ngram.cols(); j++)
					{
						int output = ngram(ngram_size-1, j);
						log_probs(0, j) = weight * output_probs(output, j);
					}
				}
				else
				{
					for (int j=0; j<ngram.cols(); j++)
					{
						int output = ngram(ngram_size-1, j);
						log_probs(0, j) = weight * prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, output, j);
					}
				}
			}

		double lookup_ngram(const int *ngram_a, int n)
		{
			for (int i=0; i<ngram_size; i++)
			{
				if (i-ngram_size+n < 0)
				{
					if (ngram_a[0] == start)
						ngram(i) = start;
					else
						ngram(i) = null;
				}
				else
				{
					ngram(i) = ngram_a[i-ngram_size+n]; //copy ngram_a[i-ngram_size+n]...ngram_a[i-1+n] to ngram
				}
			}
			return lookup_ngram(ngram);
		}

		double lookup_ngram(const std::vector<int> &ngram_v)
		{
			return lookup_ngram(ngram_v.data(), ngram_v.size());
		}

		int get_order() const { return ngram_size; }

		void read(const std::string &filename) //load the input and output vocabulary
		{
			std::vector<std::string> input_words;
			std::vector<std::string> output_words;
			nn.read(filename, input_words, output_words);
			set_input_vocabulary(vocabulary(input_words));
			set_output_vocabulary(vocabulary(output_words));
			resize();
			// this is faster but takes more memory
			nn.premultiply();
		}

		void set_cache(std::size_t cache_size)
		{
			this->cache_size = cache_size;
			cache_keys.resize(ngram_size, cache_size);
			cache_keys.fill(-1); // clears cache
			cache_values.resize(cache_size);
			cache_lookups = cache_hits = 0;
		}

		double cache_hit_rate()
		{
			return static_cast<double>(cache_hits)/cache_lookups;
		}

	};

	template <typename T>
		void addStartStop(std::vector<T> &input, std::vector<T> &output, int ngram_size, const T &start, const T &stop)  //add ngram_size-1 <s>, then input, then add the </s> at the end 
		{
			output.clear();
			output.resize(input.size()+ngram_size);
			for (int i=0; i<ngram_size-1; i++)
				output[i] = start;
			std::copy(input.begin(), input.end(), output.begin()+ngram_size-1);
			output[output.size()-1] = stop;
		}

	template <typename T>
		void makeNgrams(const std::vector<T> &input, std::vector<std::vector<T> > &output, int ngram_size)
		{
			output.clear();
			for (int j=ngram_size-1; j<input.size(); j++)
			{
				std::vector<T> ngram(input.begin() + (j-ngram_size+1), input.begin() + j+1); //copy each ngram in input to ngram
				output.push_back(ngram);
			}
		}

	template <typename T>
		void makeNgrams(const std::vector<T> &source_nums, const std::vector<T> &target_nums, const std::vector<int> &alignment_index_vector, std::vector<std::vector<T> > &input_ngrams, int source_context_size, int ngram_size, const T source_start, const T source_stop, const T target_start)
		{
			input_ngrams.clear();

			//add source_context_size source_start and source_stop
			std::vector<T> source_nums_copy;
			for (int i=0; i<source_context_size; i++) {
				source_nums_copy.push_back(source_start);
			}
			source_nums_copy.insert(source_nums_copy.end(), source_nums.begin(), source_nums.end());
			for (int i=0; i<source_context_size; i++) {
				source_nums_copy.push_back(source_stop);
			}

			for (int j=0; j<target_nums.size(); j++) {
				std::vector<T> nums;
				int aligned_source_index = alignment_index_vector[j]; //aligned source word

				//collect source-side context
				int source_begin_index = aligned_source_index - source_context_size;
				int source_end_index = aligned_source_index + source_context_size;
				for (int k=source_begin_index; k<=source_end_index; k++) { //source_context_size*2+1 source words
					nums.push_back(source_nums_copy[k+source_context_size]);
				}

				//collect target-side ngram_size-1 history
				int target_begin_index = j - ngram_size;
				for (int k=target_begin_index; k<j; k++) {
					if ( k<0 ) {
						nums.push_back(target_start);
					}
					else {
						nums.push_back(target_nums[k]);
					}
				}

				input_ngrams.push_back(nums);
			}
		}

	inline void preprocessWords(const std::vector<std::string> &words, std::vector< std::vector<int> > &ngrams,
			int ngram_size, const vocabulary &vocab, 
			bool numberize, bool add_start_stop, bool ngramize)
	{
		int start = vocab.lookup_word("<s>");
		int stop = vocab.lookup_word("</s>");

		// convert words to ints
		std::vector<int> nums;
		if (numberize) { //use user-defined vocabulary
			for (int j=0; j<words.size(); j++) {
				nums.push_back(vocab.lookup_word(words[j]));
			}
		}
		else {
			for (int j=0; j<words.size(); j++) {
				nums.push_back(boost::lexical_cast<int>(words[j])); //use boost
			}            
		}

		// convert sequence to n-grams
		ngrams.clear();
		if (ngramize) {
			std::vector<int> snums;
			if (add_start_stop) { //each line add ngram_size-1 <s> and one </s>
				addStartStop<int>(nums, snums, ngram_size, start, stop);
			} else {
				snums = nums;
			}
			makeNgrams(snums, ngrams, ngram_size);
		}
		else { //each line is a ngram
			if (nums.size() != ngram_size)
			{
				std::cerr << "error: wrong number of fields in line" << std::endl;
				std::exit(1);
			}
			ngrams.push_back(nums);
		}
	}

} // namespace nplm

#endif
