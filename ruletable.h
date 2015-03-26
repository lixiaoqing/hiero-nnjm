#include "stdafx.h"
#include "cand.h"

struct TgtRule
{
	bool operator<(const TgtRule &rhs) const{return score<rhs.score;};
	short int rule_type; 						// 规则类型，0和1表示包含0或1个非终结符，2表示包含两个正序非终结符，3表示包含两个逆序非终结符
	int word_num;                               // 规则目标端的终结符（单词）数
	vector<int> wids;                           // 规则目标端的符号（包括终结符和非终结符）id序列
	double score;                               // 规则打分, 即翻译概率与词汇权重的加权
	vector<double> probs;                       // 翻译概率和词汇权重
};

struct RuleTrieNode 
{
	vector<TgtRule> tgt_rules;                  // 一个规则源端对应的所有目标端
	map <int, RuleTrieNode*> id2subtrie_map;    // 当前规则节点到下个规则节点的转换表
};

class RuleTable
{
	public:
		RuleTable(const size_t size_limit,const Weight &i_weight,const string &rule_table_file)
		{
			RULE_NUM_LIMIT=size_limit;
			weight=i_weight;
			root=new RuleTrieNode;
			load_rule_table(rule_table_file);
		};
		vector<vector<TgtRule>* > find_matched_rules_for_prefixes(const vector<int> &src_wids,const size_t pos);

	private:
		void load_rule_table(const string &rule_table_file);
		void add_rule_to_trie(const vector<int> &src_wids, const TgtRule &tgt_rule);

	private:
		int RULE_NUM_LIMIT;                      // 每个规则源端最多加载的目标端个数 
		RuleTrieNode *root;                      // 规则Trie树根节点
		Weight weight;                           // 特征权重
};
