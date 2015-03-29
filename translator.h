#include "stdafx.h"
#include "cand.h"
#include "vocab.h"
//#include "ruletable.h"
#include "lm.h"
#include "myutils.h"

struct Models
{
	Vocab *src_vocab;
	Vocab *tgt_vocab;
	RuleTable *ruletable;
	LanguageModel *lm_model;
};

struct PatternSpan
{
	pair <int,int> span;                    //该pattern占据的span
	pair <int,int> span_src_x1;             //该pattern第一个非终结符占据的span
	pair <int,int> span_src_x2;             //该pattern第二个非终结符占据的span
};

struct Pattern
{
	vector<int> src_ids;                    //该pattern源端符号的id序列
	vector<PatternSpan> pattern_spans;      //个pattern适用的所有span
};

class SentenceTranslator
{
	public:
		SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen);
		~SentenceTranslator();
		string translate_sentence();
		vector<TuneInfo> get_tune_info(size_t sen_id);
		vector<string> get_applied_rules(size_t sen_id);
	private:
		void fill_span2cands_with_phrase_rules();
		void get_applicable_rules_for_each_span();
		void generate_kbest_for_span(const size_t beg,const size_t span);
		void get_patterns_with_one_terminal_seq(vector<Pattern> &possible_patterns);
		void get_patterns_with_two_terminal_seq(vector<Pattern> &possible_patterns);
		void get_patterns_for_glue_rule(vector<Pattern> &possible_patterns);
		void generate_cand_with_rule_and_add_to_pq(Rule &rule,int rank_x1,int rank_x2,Candpq &new_cands_by_mergence);
		void add_neighbours_to_pq(Cand *cur_cand, Candpq &new_cands_by_mergence);
		void dump_rules(vector<string> &applied_rules, Cand *cand);
		string words_to_str(vector<int> wids, bool drop_unk);

	private:
		Vocab *src_vocab;
		Vocab *tgt_vocab;
		RuleTable *ruletable;
		LanguageModel *lm_model;
		Parameter para;
		Weight feature_weight;

		vector<vector<CandBeam> > span2cands;		    //存储解码过程中所有跨度对应的候选列表, 
													    //span2cands[i][j]存储起始位置为i, 跨度为j的候选列表
		vector<vector<vector<Rule> > > span2rules;	    //存储每个跨度所有能用的hiero规则

		vector<int> src_wids;
		size_t src_sen_len;
		int src_nt_id;                                  //源端非终结符的id
		int tgt_nt_id; 									//目标端非终结符的id
};
