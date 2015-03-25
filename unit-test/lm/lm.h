#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <math.h>
using namespace std;

struct NgramTrieNode  //word list are stored in reversed order
{
	double bow;  //bow for the word list from root to current node
	map<int,double> probs;  //p(last word|history),where history is the above word list
	map<int,NgramTrieNode*> id2chilren_map;
};

class LanguageModel
{
	public:
		LanguageModel(int lm_order_in_mt,const string &lm_file){LM_ORDER_IN_MT=lm_order_in_mt;root=new NgramTrieNode;load_lm(lm_file);};
		void load_lm(const string &lm_file);
	private:
		NgramTrieNode* search_matched_path(vector<int> &word_id_list);
		void add_bow_to_trie(vector<int> &word_id_list,double bow);
		void add_prob_to_trie(vector<int> &word_id_list,double prob);

	private:
		//Vocab *tgt_vocab;
		NgramTrieNode *root;
		int LM_ORDER_IN_MT;
};
