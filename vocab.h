#ifndef VOCAB_H
#define VOCAB_H

#include "stdafx.h"

class Vocab
{
	public:
		Vocab(const string &vocab_file) {load_vocab(vocab_file);};
		string get_word(int id){return word_list.at(id);};
		int get_id(const string &word);
	private:
		void load_vocab(const string &vocab_file);
	private:
		vector<string> word_list;
		unordered_map<string,int> word2id;
};

#endif
