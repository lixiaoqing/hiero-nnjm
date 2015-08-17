#ifndef SOFTMAXLOSS_H
#define SOFTMAXLOSS_H

#include <Eigen/Dense>
#include "multinomial.h"
#include "util.h"

namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Dynamic;

///// Softmax layer plus log-loss function.

enum loss_function_type { LogLoss, NCELoss, InvalidLoss }; //only allow log-likelihood loss and noise contrastive estimation loss

inline loss_function_type string_to_loss_function (const std::string &s) //string to id
{
    if (s == "log")
        return LogLoss;
    else if (s == "nce")
        return NCELoss;
    else
        return InvalidLoss;
}

inline std::string loss_function_to_string (loss_function_type f) //id to string
{
    if (f == LogLoss)
        return "log";
    else if (f == NCELoss)
        return "nce";
}

/// Note: Outputs log-probabilities.

struct SoftmaxLogLoss
{
	template <typename DerivedI, typename DerivedW, typename DerivedO>
		void fProp(const MatrixBase<DerivedI> &input, const MatrixBase<DerivedW> &output_words, const MatrixBase<DerivedO> &output_const, double &loss) //given input, output words, obtain the output and log-likelihood loss
		{
			UNCONST(DerivedO, output_const, output);

			double log_likelihood = 0.0;

			#pragma omp parallel for reduction(+:log_likelihood)
			for (int train_id = 0; train_id < input.cols(); train_id++) //for each training instance?
			{
				double normalization = logsum(input.col(train_id)); //obtain the normalization
				output.col(train_id).array() = input.col(train_id).array() - normalization;
				log_likelihood += output(output_words(train_id), train_id); //training instance and the training word to get the likelihood
			}
			loss = log_likelihood;
		}

	template <typename DerivedW, typename DerivedO, typename DerivedI>
		void bProp(const MatrixBase<DerivedW> &output_words, const MatrixBase<DerivedO> &output, const MatrixBase<DerivedI> &grad_input_const)
		{
			UNCONST(DerivedI, grad_input_const, grad_input);
			grad_input.setZero();
			#pragma omp parallel for
			for (int train_id = 0; train_id < output.cols(); train_id++)
			{
				grad_input(output_words(train_id), train_id) += 1.; //loss=-log{p(train_word)} --> derivative_loss_train_word = {-1/p(train_word)}*derivative_p(train_word) = {-1/p(train_word)}*p(train_word)*(1-p(train_word)) = -1 + p(train_word)
				//derivative_loss_nontrain_word = {-1/p(train_word)}*{-p(train_word)*p(nontrain_word)} = p(nontrain_word)
				grad_input.col(train_id) -= output.col(train_id).array().exp().matrix();
			}
		}
};

///// Softmax layer plus NCE loss function.

///// Note: Outputs probabilities.

///// Note: Unlike SoftmaxLogLoss, does not compute *or* apply precomputed
///// normalizations. Currently the caller is expected to do normalization.

template <typename Multinomial>
class SoftmaxNCELoss
{
	const Multinomial &unigram;

	public:
	SoftmaxNCELoss(const Multinomial &unigram) 
		: unigram(unigram)
	{
	}

	template <typename DerivedI, typename DerivedW, typename DerivedO>
		void fProp(const MatrixBase<DerivedI> &scores, 
				const MatrixBase<DerivedW> &minibatch_samples,
				const MatrixBase<DerivedO> &output_const, double &loss)
		{
			UNCONST(DerivedO, output_const, output);
			double log_likelihood = 0.0;
			int num_noise_samples = minibatch_samples.rows()-1;  //noisy samples, 0 denotes true data
			double log_num_noise_samples = std::log(num_noise_samples);
			#pragma omp parallel for reduction(+:log_likelihood) schedule(static)
			for (int train_id = 0; train_id < scores.cols(); train_id++)
			{
				for (int sample_id = 0;sample_id < minibatch_samples.rows(); sample_id++)
				{
					int sample = minibatch_samples(sample_id, train_id);
					// To avoid zero or infinite probabilities,
					// never take exp of score without normalizing first,
					// even if it's a little slower...
					double score = scores(sample_id, train_id);  //prob of true data in log
					double score_noise = log_num_noise_samples + unigram.logprob(sample); //prob of noisy data
					double z = logadd(score, score_noise); //normalization
					double logprob = score - z;  //normalized prob of true data in log
					double logprob_noise = score_noise - z; //normalized prob of noisy data in log
					output(sample_id, train_id) = std::exp(logprob);
					log_likelihood += sample_id == 0 ? logprob : logprob_noise; //sample from true data or from noisy data ?
				}
			}
			loss = log_likelihood;
		}

	template <typename DerivedO, typename DerivedI>
		void bProp(const MatrixBase<DerivedO> &probs, const MatrixBase<DerivedI> &output_const)
		{
			UNCONST(DerivedI, output_const, output);
			#pragma omp parallel for schedule(static)
			for (int train_id = 0; train_id < probs.cols(); train_id++)
			{
				output.col(train_id) = -probs.col(train_id); //normalized prob of true data
				output(0, train_id) += 1.0; //normalized prob of noisy data
			}
		}
};

} // namespace nplm

#endif
