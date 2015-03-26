#ifndef DATASTRUCT_H
#define DATASTRUCT_H
#include "stdafx.h"
#include "lm/left.hh"

//生成候选所使用的规则信息
struct Rule
{
	vector<int> src_ids;      //规则源端符号（包括终结符和非终结符）id序列
	vector<int> tgt_ids;      //规则目标端符号（包括终结符和非终结符）id序列
	pair<int,int> span_x1;    //用来表示规则目标端第一个非终结符在源端的起始位置和跨度长度
	pair<int,int> span_x2;    //同上
};

//存储翻译候选
struct Cand	                
{
	//源端信息
	int rule_num;				//生成当前候选所使用的规则数目
	int glue_num;				//生成当前候选所使用的glue规则数目

	//目标端信息
	int tgt_word_num;			//当前候选目标端的单词数
	vector<int> tgt_wids;		//当前候选目标端的id序列

	//打分信息
	double score;				//当前候选的总得分
	vector<double> trans_probs;	//翻译概率
	double lm_prob;

	//合并信息,记录通过规则生成当前候选时的相关信息，注意可能只有一个子候选
	Rule applied_rule;          //生成当前候选所使用的规则源端
	int rank_x1;				//记录用的x1中的第几个候选，x1为目标端第一个非终结符
	int rank_x2;				//记录用的x2中的第几个候选
	Cand* child_x1; 			//指向改写x1的候选的指针
	Cand* child_x2;			    //指向改写x2的候选的指针

	//语言模型状态信息
	lm::ngram::ChartState lm_state;

	Cand ()
	{
		rule_num = 1;
		glue_num = 0;

		tgt_word_num = 1;
		tgt_wids.clear();

		score = 0.0;
		trans_probs.clear();
		lm_prob = 0.0;

		rank_x1 = 0;
		rank_x2 = 0;

		child_x1 = NULL;
		child_x2 = NULL;
	}
};

struct smaller
{
	bool operator() ( const Cand *pl, const Cand *pr )
	{
		return pl->score < pr->score;
	}
};

bool larger( const Cand *pl, const Cand *pr );

//将跨度相同的候选组织到列表中
class CandBeam
{
	public:
		bool add(Cand *&cand_ptr);
		Cand* top() { return data.front(); }
		Cand* at(size_t i) { return data.at(i);}
		int size() { return data.size();  }
		void sort() { std::sort(data.begin(),data.end(),larger); }
		void free();
	private:
		bool is_bound_same(const Cand *a, const Cand *b);

	private:
		vector<Cand*> data;
		vector<Cand*> recombined_cands;
};

typedef priority_queue<Cand*, vector<Cand*>, smaller> Candpq;

#endif
