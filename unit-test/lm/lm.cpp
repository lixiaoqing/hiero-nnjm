#include "lm.h"

void LanguageModel::load_lm(const string &lm_file)
{
	ifstream fin(lm_file.c_str(),ios::binary);
	if (!fin.is_open())
	{
		cerr<<"cannot open language model file!\n";
		return;
	}
	cout<<"loading language model file: "<<lm_file<<endl;

	int max_order;
	fin.read((char*)&max_order,sizeof(int));
	vector<int> num_for_each_ngram;
	num_for_each_ngram.resize(max_order);
	fin.read((char*)&num_for_each_ngram[0],sizeof(int)*max_order);
	for(size_t i=0;i<max_order;i++)
	{
		for(size_t j=0;j<num_for_each_ngram.at(i);j++)
		{
			double prob,bow;
			size_t order= i+1;
			vector<int> word_id_list;
			word_id_list.resize(order);
			fin.read((char*)&prob,sizeof(double));
			fin.read((char*)&word_id_list[0],sizeof(int)*(order));
			reverse(word_id_list.begin(),word_id_list.end());
			//append VocabNone to the end of word_id_list??
			if(order != max_order)
			{
				fin.read((char*)&bow,sizeof(double));
			}
			if(order<max_order && bow != 0.0)
			{
				add_bow_to_trie(word_id_list,bow);
			}
			add_prob_to_trie(word_id_list,prob);
		}
	}
}

NgramTrieNode* LanguageModel::search_matched_path(vector<int> &word_id_list)
{
	NgramTrieNode* current = root;
	for (const auto &word_id : word_id_list)
	{        
		auto it = current->id2chilren_map.find(word_id);
		if ( it != current->id2chilren_map.end() )
		{
			current = it->second;
		}
		else
		{
			NgramTrieNode* tmp = new NgramTrieNode();
			current->id2chilren_map.insert(make_pair(word_id,tmp));
			current = tmp;
		}
	}
	return current;
}

void LanguageModel::add_bow_to_trie(vector<int> &word_id_list,double bow)
{
	NgramTrieNode* node=search_matched_path(word_id_list);
	node->bow=bow;
}

void LanguageModel::add_prob_to_trie(vector<int> &word_id_list,double prob)
{
	vector<int> history(word_id_list.begin()+1,word_id_list.end());
	NgramTrieNode* node=search_matched_path(history);
	node->probs.insert(make_pair(word_id_list.at(0),prob));
}

int main()
{
	LanguageModel lm(3,"lm.bin");
}
