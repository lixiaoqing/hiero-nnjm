#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>

using namespace std;
using namespace TCLAP;

#include "neuralLM.h" // for vocabulary
#include "util.h"

using namespace boost;
using namespace nplm;

void transformAlignment(const vector<string> &alignment_string_vector, vector<int> &alignment_index_vector)
{
		unordered_map<int,vector<int> > target_alignment_map; //alignment string to id first
		for(int i = 0; i < alignment_string_vector.size(); i++) {
			string source_target_index_pair = alignment_string_vector[i];
			size_t split_pos = source_target_index_pair.find("-");
			string source_index_str = source_target_index_pair.substr(0, split_pos);
			string target_index_str = source_target_index_pair.substr(split_pos+1, -1);
			int source_index = atoi(source_index_str.c_str()), target_index = atoi(target_index_str.c_str());
			unordered_map<int, vector<int> >::iterator alignment_iter = target_alignment_map.find(target_index);
			if( alignment_iter == target_alignment_map.end()) {
				vector<int> source_index_vector;
				source_index_vector.push_back(source_index);
				target_alignment_map.insert( make_pair(target_index,  source_index_vector) );
			}
			else {
				alignment_iter->second.push_back(source_index);
			}
		}

		//according to the alignment and affiliation heuristic, keep exactly one source index for each target index
		for( int i = 0; i < alignment_index_vector.size(); i++ ) {
			unordered_map<int, vector<int> >::iterator current_alignment_iter =  target_alignment_map.find(i);
			if( current_alignment_iter != target_alignment_map.end() ) {
				int source_words_num = current_alignment_iter->second.size();
				if( 1 == source_words_num ) { //the target word exactly aligns to one source word
					alignment_index_vector[i] = current_alignment_iter->second[0];
				}
				else {
					sort(current_alignment_iter->second.begin(), current_alignment_iter->second.end());
					alignment_index_vector[i] = current_alignment_iter->second[source_words_num/2]; //choose the middle one
				}
			}
			else { //no alignment source words, choose the nearest one, from right to left
				int distance = 1;
				int target_len = alignment_index_vector.size();
				int right_index = i + distance, left_index = i - distance;
				while( right_index < target_len || left_index >= 0 ) { //not exceed the sentence boundary
					if( right_index < target_len ) { //from right to left
						unordered_map<int, vector<int> >::iterator right_iter = target_alignment_map.find(right_index);
						if( right_iter != target_alignment_map.end() ) {
							int source_words_num = right_iter->second.size();
							if( 1 == source_words_num ) {
								alignment_index_vector[i] = right_iter->second[0];
								break;
							}
							else {
								sort(right_iter->second.begin(), right_iter->second.end());
								alignment_index_vector[i] = right_iter->second[source_words_num/2];
								break;
							}
						}
					}

					if( left_index >= 0 ) {
						unordered_map<int, vector<int> >::iterator left_iter = target_alignment_map.find(left_index);
						if( left_iter != target_alignment_map.end() ) {
							int source_words_num = left_iter->second.size();
							if( 1 == source_words_num ) {
								alignment_index_vector[i] = left_iter->second[0];
								break;
							}
							else {
								sort(left_iter->second.begin(), left_iter->second.end());
								alignment_index_vector[i] = left_iter->second[source_words_num/2];
								break;
							}
						}
					}

					distance += 1;
					right_index = i + distance, left_index = i - distance;
				}

				if( right_index >= target_len && left_index < 0 ) { 
					alignment_index_vector[i] = 0;
				}
			}
		}
}

void writeNgrams(const vector<vector<string> > &input_data, const vector<vector<string> > &output_data, const vector<vector<string> > &alignment_data, int source_context_size, int ngram_size, const vocabulary &input_vocab, const vocabulary &output_vocab, bool numberize, bool ngramize, const string &filename) //prepare training instances
{
	ofstream file(filename.c_str());
	if (!file)
	{
		cerr << "error: could not open " << filename << endl;
		exit(1);
	}

	// check that input, output and alignment data have the same number of sentences
	if (input_data.size() != output_data.size() || input_data.size() != alignment_data.size()) {
		cerr << "Error: input and output data files have different number of lines" << endl;
		exit(1);
	}

	int source_start = input_vocab.lookup_word("<src>");
	int source_stop = input_vocab.lookup_word("</src>");
	int target_start = input_vocab.lookup_word("<tgt>");
	// for each input and output line
	int lines=input_data.size();
	if (numberize) { //convert string to index
		for (int i=0; i<lines; i++) {
			//transform the alignment string first
			vector<int> alignment_index_vector(output_data[i].size());
			transformAlignment(alignment_data[i], alignment_index_vector);

			// convert each line to a set of ngrams: source_context + target_histroy_ngram
			vector<vector<int> > input_ngrams;
			vector<int> source_nums;
			for (int j=0; j<input_data[i].size(); j++) {
				source_nums.push_back(input_vocab.lookup_word(input_data[i][j]));
			}
			vector<int> target_nums;
			for (int j=0; j<output_data[i].size(); j++) {
				target_nums.push_back(input_vocab.lookup_word(output_data[i][j]));
			}

			makeNgrams(source_nums, target_nums, alignment_index_vector, input_ngrams, source_context_size, ngram_size-1, source_start, source_stop, target_start);

			vector<vector<int> > output_ngrams;
			vector<int> output_nums;
			for (int j=0; j<output_data[i].size(); j++) {
				output_nums.push_back(output_vocab.lookup_word(output_data[i][j]));
			}
			makeNgrams(output_nums, output_ngrams, 1);

			// print out source and target contexts ngrams
			if (input_ngrams.size() != output_ngrams.size()) {
				cerr<<"input ngrams: "<<input_ngrams.size()<<"  are not consistant with the outputs "<<output_ngrams.size()<<"!"<<endl;
			}

			for (int j = 0; j < input_ngrams.size(); j++ ) {
				file << i << " "; //sentence id for indexing the sentence embeddings
				for (int k = 0; k < input_ngrams[j].size(); k++) {
					file << input_ngrams[j][k] << " ";
				}
				file << output_ngrams[j][0] << endl;
			}
		}
	}

	else {
		for (int i=0; i<lines; i++) {
			//transform the alignment string first
			vector<int> alignment_index_vector(output_data[i].size());
			transformAlignment(alignment_data[i], alignment_index_vector);

			// convert each line to a set of ngrams: source_context + target_histroy_ngram
			vector<vector<string> > input_ngrams;
			vector<string> source_words;
			for (int j=0; j<input_data[i].size(); j++) {
				int unk = input_vocab.lookup_word("<unk>");
				//if word is unknown
				if (input_vocab.lookup_word(input_data[i][j]) == unk) {
					source_words.push_back("<unk>");
				}
				//if word is known
				else {
					source_words.push_back(input_data[i][j]);
				}
			}
			vector<string> target_words;
			for (int j=0; j<output_data[i].size(); j++) {
				int unk = input_vocab.lookup_word("<unk>");
				if( input_vocab.lookup_word(output_data[i][j]) == unk ) {
					target_words.push_back("<unk>");
				}
				else {
					target_words.push_back(output_data[i][j]);
				}
			}

			string source_start_str = "<src>", source_stop_str = "</src>", target_start_str = "<tgt>";
			makeNgrams(source_words, target_words, alignment_index_vector, input_ngrams, source_context_size, ngram_size-1, source_start_str, source_stop_str, target_start_str);

			vector<vector<string> > output_ngrams;
			vector<string> output_words;
			for (int j=0; j<output_data[i].size(); j++) {
				int unk = output_vocab.lookup_word("<unk>");
				// if word is unknown
				if (output_vocab.lookup_word(output_data[i][j]) == unk) {
					output_words.push_back("<unk>");
				}
				// if word is known
				else {
					output_words.push_back(output_data[i][j]);
				}
			}
			makeNgrams(output_words, output_ngrams, 1);

			// print out source and target contexts ngrams
			if (input_ngrams.size() != output_ngrams.size()) {
				cerr<<"input ngrams: "<<input_ngrams.size()<<"  are not consistant with the outputs "<<output_ngrams.size()<<"!"<<endl;
			}

			for (int j = 0; j < input_ngrams.size(); j++ ) {
				file << i << " "; //sentence id for indexing the sentence embeddings
				for (int k = 0; k < input_ngrams[j].size(); k++) {
					file << input_ngrams[j][k] << " ";
				}
				file << output_ngrams[j][0] << endl;
			}
		}
	}
	file.close();
}
    
int main(int argc, char *argv[])
{
	int ngram_size, source_context_size, input_vocab_size, output_vocab_size, validation_size;
	bool numberize, ngramize;
	string input_train_text, output_train_text, alignment_train_text, train_file, input_validation_text, output_validation_text, alignment_validation_text, validation_file, write_input_words_file, write_output_words_file, input_words_file, output_words_file;

	try
	{
		CmdLine cmd("Prepares training data for training a language model.", ' ', "0.1");

		// The options are printed in reverse order

		ValueArg<bool> arg_ngramize("", "ngramize", "If true, convert lines to ngrams. Default: true.", false, true, "bool", cmd);
		ValueArg<bool> arg_numberize("", "numberize", "If true, convert words to numbers. Default: true.", false, true, "bool", cmd);
		ValueArg<int> arg_input_vocab_size("", "input_vocab_size", "Vocabulary size.", false, -1, "int", cmd);
		ValueArg<int> arg_output_vocab_size("", "output_vocab_size", "Vocabulary size.", false, -1, "int", cmd);
		ValueArg<string> arg_input_words_file("", "input_words_file", "File specifying words that should be included in vocabulary; all other words will be replaced by <unk>.", false, "", "string", cmd);
		ValueArg<string> arg_output_words_file("", "output_words_file", "File specifying words that should be included in vocabulary; all other words will be replaced by <unk>.", false, "", "string", cmd);
		ValueArg<int> arg_ngram_size("", "ngram_size", "Size of n-grams.", true, -1, "int", cmd);
		ValueArg<int> arg_source_context_size("", "source_context_size", "Size of source context.", true, -1, "int", cmd);
		ValueArg<string> arg_write_input_words_file("", "write_input_words_file", "Output vocabulary.", false, "", "string", cmd);
		ValueArg<string> arg_write_output_words_file("", "write_output_words_file", "Output vocabulary.", false, "", "string", cmd);
		ValueArg<int> arg_validation_size("", "validation_size", "How many lines from training data to hold out for validation. Default: 0.", false, 0, "int", cmd);
		ValueArg<string> arg_validation_file("", "validation_file", "Output validation data (numberized n-grams).", false, "", "string", cmd);
		ValueArg<string> arg_input_validation_text("", "input_validation_text", "Input validation data (tokenized). Overrides --validation_size. Default: none.", false, "", "string", cmd);
		ValueArg<string> arg_output_validation_text("", "output_validation_text", "Input validation data (tokenized). Overrides --validation_size. Default: none.", false, "", "string", cmd);
		ValueArg<string> arg_alignment_validation_text("", "alignment_validation_text", "Input validation data (tokenized). Overrides --validation_size. Default: none.", false, "", "string", cmd);
		ValueArg<string> arg_train_file("", "train_file", "Output training data (numberized n-grams).", false, "", "string", cmd);
		ValueArg<string> arg_input_train_text("", "input_train_text", "Input training data (tokenized).", true, "", "string", cmd);
		ValueArg<string> arg_output_train_text("", "output_train_text", "Input training data (tokenized).", true, "", "string", cmd);
		ValueArg<string> arg_alignment_train_text("", "alignment_train_text", "Input training data (tokenized).", true, "", "string", cmd);

		cmd.parse(argc, argv);

		input_train_text = arg_input_train_text.getValue();    //input training
		output_train_text = arg_output_train_text.getValue();  //output training
		alignment_train_text = arg_alignment_train_text.getValue();  //alignment training
		train_file = arg_train_file.getValue();                //training file for neural network
		validation_file = arg_validation_file.getValue();      //validation file for neural network
		input_validation_text = arg_input_validation_text.getValue();  //input validation
		output_validation_text = arg_output_validation_text.getValue(); //output valdation
		alignment_validation_text = arg_alignment_validation_text.getValue(); //alignment valdation
		validation_size = arg_validation_size.getValue();     //validation data size
		write_input_words_file = arg_write_input_words_file.getValue();  //input words file
		write_output_words_file = arg_write_output_words_file.getValue(); //output words file
		ngram_size = arg_ngram_size.getValue();  //ngram order: 4
		source_context_size = arg_source_context_size.getValue();  //source-side context size: 5
		input_vocab_size = arg_input_vocab_size.getValue();  //input vocabulary
		output_vocab_size = arg_output_vocab_size.getValue(); //output vocabulary
		input_words_file = arg_input_words_file.getValue();  //original input words file
		output_words_file = arg_output_words_file.getValue(); //original output words file
		numberize = arg_numberize.getValue();  //whether do string to int
		ngramize = arg_ngramize.getValue();    //whether sequence to ngrams

		// check command line arguments

		// Notes:
		// - either --words_file or --vocab_size is required.
		// - if --words_file is set,
		// - if --vocab_size is not set, it is inferred from the length of the file
		// - if --vocab_size is set, it is an error if the vocab file has a different number of lines
		// - if --numberize 0 is set and --use_vocab f is not set, then the output model file will not have a vocabulary, and a warning should be printed.
		if ((input_words_file == "") && (input_vocab_size == -1)) {
			cerr << "Error: either --input_words_file or --input_vocab_size is required." << endl;
			exit(1);
		}
		if ((output_words_file == "") && (output_vocab_size == -1)) {
			cerr << "Error: either --output_words_file or --output_vocab_size is required." << endl;
			exit(1);
		}

		// Notes:
		// - if --ngramize 0 is set, then
		// - if --ngram_size is not set, it is inferred from the training file (different from current)
		// - if --ngram_size is set, it is an error if the training file has a different n-gram size
		// - if neither --validation_file or --validation_size is set, validation will not be performed.
		// - if --numberize 0 is set, then --validation_size cannot be used.

		cerr << "Command line: " << endl;
		cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

		const string sep(" Value: ");
		cerr << arg_input_train_text.getDescription() << sep << arg_input_train_text.getValue() << endl;
		cerr << arg_output_train_text.getDescription() << sep << arg_output_train_text.getValue() << endl;
		cerr << arg_alignment_train_text.getDescription() << sep << arg_alignment_train_text.getValue() << endl;
		cerr << arg_train_file.getDescription() << sep << arg_train_file.getValue() << endl;
		cerr << arg_input_validation_text.getDescription() << sep << arg_input_validation_text.getValue() << endl;
		cerr << arg_output_validation_text.getDescription() << sep << arg_output_validation_text.getValue() << endl;
		cerr << arg_alignment_validation_text.getDescription() << sep << arg_alignment_validation_text.getValue() << endl;
		cerr << arg_validation_file.getDescription() << sep << arg_validation_file.getValue() << endl;
		cerr << arg_validation_size.getDescription() << sep << arg_validation_size.getValue() << endl;
		cerr << arg_write_input_words_file.getDescription() << sep << arg_write_input_words_file.getValue() << endl;
		cerr << arg_write_output_words_file.getDescription() << sep << arg_write_output_words_file.getValue() << endl;
		cerr << arg_ngram_size.getDescription() << sep << arg_ngram_size.getValue() << endl;
		cerr << arg_source_context_size.getDescription() << sep << arg_source_context_size.getValue() << endl;
		cerr << arg_input_vocab_size.getDescription() << sep << arg_input_vocab_size.getValue() << endl;
		cerr << arg_output_vocab_size.getDescription() << sep << arg_output_vocab_size.getValue() << endl;
		cerr << arg_input_words_file.getDescription() << sep << arg_input_words_file.getValue() << endl;
		cerr << arg_output_words_file.getDescription() << sep << arg_output_words_file.getValue() << endl;
		cerr << arg_numberize.getDescription() << sep << arg_numberize.getValue() << endl;
		cerr << arg_ngramize.getDescription() << sep << arg_ngramize.getValue() << endl;
	}
	catch (TCLAP::ArgException &e)
	{
		cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
		exit(1);
	}

	// Read in input training data and validation data
	vector<vector<string> > input_train_data;
	readSentFile(input_train_text, input_train_data); //read each line in training data and stroe in the input_train_data

	vector<vector<string> > input_validation_data;
	if (input_validation_text != "") {
		readSentFile(input_validation_text, input_validation_data);
	}
	else if (validation_size > 0)
	{
		if (validation_size > input_train_data.size())
		{
			cerr << "error: requested input_validation size is greater than training data size" << endl;
			exit(1);
		}
		input_validation_data.insert(input_validation_data.end(), input_train_data.end()-validation_size, input_train_data.end());
		input_train_data.resize(input_train_data.size() - validation_size);
	}

	// Read in output training data and validation data
	vector<vector<string> > output_train_data;
	readSentFile(output_train_text, output_train_data);

	vector<vector<string> > output_validation_data;
	if (output_validation_text != "") {
		readSentFile(output_validation_text, output_validation_data);
	}
	else if (validation_size > 0)
	{
		if (validation_size > output_train_data.size())
		{
			cerr << "error: requested output_validation size is greater than training data size" << endl;
			exit(1);
		}
		output_validation_data.insert(output_validation_data.end(), output_train_data.end()-validation_size, output_train_data.end());
		output_train_data.resize(output_train_data.size() - validation_size);
	}

	// Read in alignment training data and validation data
	vector<vector<string> > alignment_train_data;
	readSentFile(alignment_train_text, alignment_train_data);

	vector<vector<string> > alignment_validation_data;
	if (alignment_validation_text != "") {
		readSentFile(alignment_validation_text, alignment_validation_data);
	}
	else if (validation_size > 0)
	{
		if (validation_size > alignment_train_data.size())
		{
			cerr << "error: requested alignment_validation size is greater than training data size" << endl;
			exit(1);
		}
		alignment_validation_data.insert(alignment_validation_data.end(), alignment_train_data.end()-validation_size, alignment_train_data.end());
		alignment_train_data.resize(alignment_train_data.size() - validation_size);
	}

	// Construct input vocabulary
	vocabulary input_vocab;
	int input_source_start = input_vocab.insert_word("<src>");
	int input_source_stop = input_vocab.insert_word("</src>");
	int input_target_start = input_vocab.insert_word("<tgt>");
	int input_target_stop = input_vocab.insert_word("</tgt>");
	input_vocab.insert_word("<null>");

	int source_vocab_size = input_vocab_size/2;
	int target_vocab_size = source_vocab_size;

	unordered_map<string, int> target_counts; //count the target-side words
	for (int i = 0; i < output_train_data.size(); i++) {
		for (int j = 0; j < output_train_data[i].size(); j++) {
			target_counts[output_train_data[i][j]] += 1;
		}
	}

	// read input vocabulary from file
	if (input_words_file != "") {
		vector<string> words;
		readWordsFile(input_words_file,words);
		for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
			input_vocab.insert_word(*it);
		}
		// was input_vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
		if (input_vocab_size > 0) {
			if (input_vocab.size() != input_vocab_size) {
				cerr << "Error: size of input_vocabulary file " << input_vocab.size() << " != --input_vocab_size " << input_vocab_size << endl;
			}
		}
		// else, set it to the size of vocabulary read from file
		else {
			input_vocab_size = input_vocab.size();
		}
	}

	// or construct input vocabulary to contain top <input_vocab_size> most frequent words; all other words replaced by <unk>
	else {
		unordered_map<string,int> count;
		for (int i=0; i<input_train_data.size(); i++) {
			for (int j=0; j<input_train_data[i].size(); j++) {
				count[input_train_data[i][j]] += 1; 
			}
		}

		input_vocab.insert_most_frequent(count, source_vocab_size);
		if (input_vocab.size() < source_vocab_size) {
			cerr << "warning: fewer than " << source_vocab_size << " types in source training data; the unknown word will not be learned" << endl;
		}

		input_vocab.insert_most_frequent(target_counts, input_vocab_size);
		if (input_vocab.size() < input_vocab_size) {
			cerr << "warning: fewer than " << target_vocab_size << " types in target training data; the unknown word will not be learned" << endl;
		}

	}

	// Construct output vocabulary
	vocabulary output_vocab;
	int output_start = output_vocab.insert_word("<tgt>");
	int output_stop = output_vocab.insert_word("</tgt>");
	output_vocab.insert_word("<null>");

	// read output vocabulary from file
	if (output_words_file != "") {
		vector<string> words;
		readWordsFile(output_words_file,words);
		for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
			output_vocab.insert_word(*it);
		}
		// was output_vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
		if (output_vocab_size > 0) {
			if (output_vocab.size() != output_vocab_size) {
				cerr << "Error: size of output_vocabulary file " << output_vocab.size() << " != --output_vocab_size " << output_vocab_size << endl;
			}
		}
		// else, set it to the size of vocabulary read from file
		else {
			output_vocab_size = output_vocab.size();
		}
	}

	// or construct output vocabulary to contain top <output_vocab_size> most frequent words; all other words replaced by <unk>
	else {
		output_vocab.insert_most_frequent(target_counts, output_vocab_size);
		if (output_vocab.size() < output_vocab_size) {
			cerr << "warning: fewer than " << output_vocab_size << " types in training data; the unknown word will not be learned" << endl;
		}
	}

	// write input vocabulary to file
	if (write_input_words_file != "") {
		cerr << "Writing vocabulary to " << write_input_words_file << endl;
		writeWordsFile(input_vocab.words(), write_input_words_file);
	}

	// write output vocabulary to file
	if (write_output_words_file != "") {
		cerr << "Writing vocabulary to " << write_output_words_file << endl;
		writeWordsFile(output_vocab.words(), write_output_words_file);
	}

	// Write out input and output numberized n-grams
	if (train_file != "")
	{
		cerr << "Writing training data to " << train_file << endl;
		writeNgrams(input_train_data, output_train_data, alignment_train_data, source_context_size, ngram_size, input_vocab, output_vocab, numberize, ngramize, train_file);

	}
	if (validation_file != "")
	{
		cerr << "Writing validation data to " << validation_file << endl;
		writeNgrams(input_validation_data, output_validation_data, alignment_validation_data, source_context_size, ngram_size, input_vocab, output_vocab, numberize, ngramize, validation_file);
	}
}
