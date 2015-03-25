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

const int RULE_LEN_MAX=7;
const int END_ID=0;
const int PROB_NUM=4;
const double LogP_PseudoZero=-99;

struct TgtRule
{
	bool operator<(const TgtRule &right) const {return score < right.score;};
	int word_num;
	vector<int> word_id_list;
	double score;
	vector<double> prob_list;
	vector<vector<int> > ch_pos_to_en_pos_list;
};

struct TrieNode 
{
	vector<TgtRule> m_tgt_rule_list;
	map<int, TrieNode*> m_children;
};

void add_rule_to_trie(TrieNode *root,const vector<int> &src_rule, const TgtRule &tgt_rule)
{
	TrieNode* current = root;
	for (const auto &word_id : src_rule)
	{        
		auto child = current->m_children.find(word_id);
		if ( child != current->m_children.end() )
		{
			current = child->second;
		}
		else
		{
			TrieNode* tmp = new TrieNode();
			current->m_children.insert(make_pair(word_id,tmp));
			current = tmp;
		}
	}
	if (current->m_tgt_rule_list.size() < 50)
	{
		current->m_tgt_rule_list.push_back(tgt_rule);
	}
	else
	{
		auto it = min_element(current->m_tgt_rule_list.begin(), current->m_tgt_rule_list.end());
		if( it->score < tgt_rule.score )
		{
			(*it) = tgt_rule;
		}
	}
}

int main()
{
	TrieNode *m_rule_table = new TrieNode;
	ifstream fin("prob.bin",ios::binary);
	if (!fin.is_open())
	{
		cerr<<"cannot open rule table file!\n";
	}
	cout<<"loading rule table file\n";
	short int src_rule_len=0;
	while(fin.read((char*)&src_rule_len,sizeof(short int)))
	{
		vector<int> src_rule;
		src_rule.resize(src_rule_len+1);
		fin.read((char*)&src_rule[0],sizeof(int)*src_rule_len);
		src_rule[src_rule_len] = END_ID;

		/*
		cout<<"src rule\n";
		for (auto e:src_rule)
		{
			cout<<e<<'\t';
		}
		cout<<endl;
		*/

		short int tgt_rule_len=0;
		fin.read((char*)&tgt_rule_len,sizeof(short int));
		if (tgt_rule_len > RULE_LEN_MAX)
			continue;
		TgtRule tgt_rule;
		tgt_rule.word_num = tgt_rule_len;
		tgt_rule.word_id_list.resize(tgt_rule_len+1);
		fin.read((char*)&(tgt_rule.word_id_list[0]),sizeof(int)*tgt_rule_len);
		tgt_rule.word_id_list[tgt_rule_len] = END_ID;

		/*
		cout<<"tgt rule\n";
		for (auto e:tgt_rule.word_id_list)
		{
			cout<<e<<'\t';
		}
		cout<<endl;
		*/

		//cout<<"prob\n";
		tgt_rule.prob_list.resize(PROB_NUM);
		fin.read((char*)&(tgt_rule.prob_list[0]),sizeof(double)*PROB_NUM);
		for(auto &e : tgt_rule.prob_list)
		{
			if( e == 0.0 )
			{
				e = LogP_PseudoZero;
			}
			else
			{
				e = log10(e);
			}
			//cout<<e<<'\t';
		}
		//cout<<endl;

		/*
		for(size_t i=0;i<PROB_NUM;i++)	
		{
			if( tgt_rule.prob_list[i] == 0.0 )
			{
				tgt_rule.prob_list[i] = LogP_PseudoZero;
			}
			else
			{
				tgt_rule.prob_list[i] = log10(tgt_rule.prob_list[i]);
			}
		}
		*/
		short int alignment_num=0;
		fin.read((char*)&alignment_num,sizeof(short int));
		int *alignment_array = new int[alignment_num];
		fin.read((char*)alignment_array,sizeof(int)*alignment_num);

		//cout<<"alignment\n";
		tgt_rule.ch_pos_to_en_pos_list.resize(src_rule_len);
		for(size_t i=0;i<alignment_num/2;i++)
		{
			int ch_pos = alignment_array[2*i];
			int en_pos = alignment_array[2*i+1];
			tgt_rule.ch_pos_to_en_pos_list[ch_pos].push_back(en_pos);
			//cout<<ch_pos<<'\t'<<en_pos<<endl;
		}


		tgt_rule.score = 0;
		if( tgt_rule.prob_list.size() != 4 )
		{
			cout<<"number of probability in rule is wrong!"<<endl;
		}
		for( size_t i=0; i<4; i++ )
		{
			tgt_rule.score += tgt_rule.prob_list[i]*1;
		}
		//cin.get();

		add_rule_to_trie(m_rule_table,src_rule,tgt_rule);

	}
	cout<<"loading over\n";
	//cin.get();
	fin.close();
}
