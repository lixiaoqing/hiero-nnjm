#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>
#include <string>
#include <Eigen/Dense>

#include "util.h"

namespace nplm
{

	// is this cheating?
	using Eigen::Matrix;
	using Eigen::MatrixBase;

	enum activation_function_type { Tanh, HardTanh, Rectifier, Identity, InvalidFunction }; //four activation types supported

	inline activation_function_type string_to_activation_function (const std::string &s) //string to id
	{
		if (s == "identity")
			return Identity;
		else if (s == "rectifier")
			return Rectifier;
		else if (s == "tanh")
			return Tanh;
		else if (s == "hardtanh")
			return HardTanh;
		else
			return InvalidFunction;
	}

	inline std::string activation_function_to_string (activation_function_type f)    //id to string
	{
		if (f == Identity)
			return "identity";
		else if (f == Rectifier)
			return "rectifier";
		else if (f == Tanh)
			return "tanh";
		else if (f == HardTanh)
			return "hardtanh";
	}

	//activation function: hardtanh [-1, 1] x < -1 f(x)=-1; x > 1 f(x)=1 else f(x)=x 
	struct hardtanh_functor {
		double operator() (double x) const { if (x < -1.) return -1.; else if (x > 1.) return 1.; else return x; }
	};

	//derivative of hardtanh function: x in [-1, 1] x'=1
	struct dhardtanh_functor {
		double operator() (double x) const { return x > -1. && x < 1. ? 1. : 0.; }
	};

	//activation function: tanh(.)
	struct tanh_functor {
		double operator() (double x) const { return std::tanh(x); }
	};

	//derivative of tanh function: x'=1-x^2
	struct dtanh_functor {
		double operator() (double x) const { return 1-x*x; }
	};

	//activation function: rectifier x<0 f(x)=0; x>0 f(x)=x
	struct rectifier_functor {
		double operator() (double x) const { return std::max(x, 0.); }
	};

	//derivative of rectifier: x>0 x'=1; else x'=0
	struct drectifier_functor {
		double operator() (double x) const { return x > 0. ? 1. : 0.; }
	};

	class Activation_function
	{
		private:
			int size;                 //input and output size
			activation_function_type f;   //activation type

		public:
			Activation_function() : size(0), f(Rectifier) { }  //default use rectifier activation function

			void resize(int size) { this->size = size; }
			void set_activation_function(activation_function_type f) { this->f = f; }

			template <typename Engine>
				void initialize(Engine &engine, bool init_normal, double init_range) { }

			int n_inputs () const { return size; }
			int n_outputs () const { return size; }

			template <typename DerivedIn, typename DerivedOut>
				void fProp(const MatrixBase<DerivedIn> &input, const MatrixBase<DerivedOut> &output) const  //forward propagation
				{
					UNCONST(DerivedOut, output, my_output);  //transform const output to non-const

					switch (f) //calculate the output according to the activation function
					{
						case Identity: my_output = input; break;
						case Rectifier: my_output = input.unaryExpr(rectifier_functor()); break;  //element-wise computation
						case Tanh: my_output = input.unaryExpr(tanh_functor()); break;
						case HardTanh: my_output = input.unaryExpr(hardtanh_functor()); break;
					}
				}

			template <typename DerivedGOut, typename DerivedGIn, typename DerivedIn, typename DerivedOut>
				void bProp(const MatrixBase<DerivedGOut> &input, MatrixBase<DerivedGIn> &output,
						const MatrixBase<DerivedIn> &finput, const MatrixBase<DerivedOut> &foutput) const //backward propagation for derivatives
				{
					UNCONST(DerivedGIn, output, my_output); //transform const output to non-const

					switch (f)
					{
						case Identity: my_output = input; break;
						case Rectifier: my_output = finput.array().unaryExpr(drectifier_functor()) * input.array(); break;
						case Tanh: my_output = foutput.array().unaryExpr(tanh_functor()) * input.array(); break;         //?
						case HardTanh: my_output = finput.array().unaryExpr(hardtanh_functor()) * input.array(); break;  //?
					}
				}
	};

} // namespace nplm

#endif
