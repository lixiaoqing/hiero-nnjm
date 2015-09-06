#pragma once

#include <string>

namespace nplm
{

struct param 
{
    std::string train_file;       //raw data file for training
    std::string validation_file;  //validation file for training
    std::string test_file;       //test file for perlexicity calcuation?

    std::string model_file;      //file storing neural network model including word embeddings

    std::string unigram_probs_file; //file storing probabilities of unigrams
    std::string words_file;         //?
    std::string input_words_file;   //raw file storing all words?
    std::string output_words_file;  //new file storing most frequent words?
    std::string model_prefix;       //?

    int ngram_size;                 //ngram order used for English, if order is 4, taking into account 3 previous words
    int vocab_size;                 //the size of vocab: the number of most frequent words
    int input_vocab_size;           //the size of vocab in the raw file
    int output_vocab_size;          //the size of vocab in the new file
    int num_hidden;                 //the number of hidden layers
    int embedding_dimension;        //the dimension of embedding each word
    int input_embedding_dimension;  //?
    int output_embedding_dimension; //?
    std::string activation_function; //the type of activation function, such as tanh, sigmoid
    std::string loss_function;       //objective function

    int minibatch_size;              //the size of each data block used in parallel computing
    int validation_minibatch_size;   //the minibatch size used to handle validation data
    int num_epochs;                  //the iteration number
    double learning_rate;            //the learning rate for parameter update

    bool init_normal;                //initialize the parameters with normal distribution or uniform distribution
    double init_range;               //interval of the intialized parameters

    int num_noise_samples;          //the number of noise samples applied in noise contrastive estimation

    bool use_momentum;              //?
    double initial_momentum;        //?
    double final_momentum;          //?

    double L2_reg;                  //L2 regulization

    bool normalization;             //normalize the output in good probability distribution or not
    double normalization_init;      //?

    int num_threads;                //the number of threads for parallel computing
  
    bool share_embeddings;         //who share embedding with who?

};

} // namespace nplm
