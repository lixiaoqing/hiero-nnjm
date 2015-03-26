#include "stdafx.h"
#include "cand.h"
#include "vocab.h"
#include "lm/model.hh"
#include "lm/left.hh"
#include "lm/enumerate_vocab.hh"
using namespace lm::ngram;

class LanguageModel
{
	public:
		LanguageModel(const string &lm_file, Vocab *tgt_vocab);
		double cal_increased_lm_score(Cand* cand);
		double cal_final_increased_lm_score(Cand* cand);

	private:
			lm::WordIndex convert_to_kenlm_id(int wid);
	private:
		Model *kenlm;
		vector<lm::WordIndex> ori_to_kenlm_id;
		lm::WordIndex EOS;
		int nonterminal_wid;
};
