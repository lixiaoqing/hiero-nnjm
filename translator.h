#include "stdafx.h"
#include "cand.h"
#include "vocab.h"
#include "ruletable.h"
#include "lm.h"
#include "myutils.h"

struct Models
{
	Vocab *src_vocab;
	Vocab *tgt_vocab;
	RuleTable *ruletable;
	LanguageModel *lm_model;
};

struct Pattern
{
	vector<int> src_ids;
	pair<int,int> span_src_x1;
	pair<int,int> span_src_x2;
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
		void fill_matrix_with_matched_rules();
		void generate_kbest_for_span(const size_t beg,const size_t span);
		void get_patterns_with_one_terminal(const size_t beg,const size_t span,vector<Pattern> &possible_patterns);
		void get_patterns_with_two_terminals(const size_t beg,const size_t span,vector<Pattern> &possible_patterns);
		void get_patterns_for_glue_rule(const size_t beg,const size_t span,vector<Pattern> &possible_patterns);
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

		vector<vector<CandBeam> > candbeam_matrix;		//存储解码过程中所有跨度对应的候选列表, 
													    //candbeam_matrix[i][j]存储起始位置为i, 跨度为j的候选列表
		vector<int> src_wids;
		size_t src_sen_len;
};
