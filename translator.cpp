#include "translator.h"

SentenceTranslator::SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen)
{
	src_vocab = i_models.src_vocab;
	tgt_vocab = i_models.tgt_vocab;
	ruletable = i_models.ruletable;
	lm_model = i_models.lm_model;
    nnjm_model = i_models.nnjm_model;
	para = i_para;
	feature_weight = i_weight;

	src_nt_id = src_vocab->get_id("[X][X]");
	tgt_nt_id = tgt_vocab->get_id("[X][X]");

    src_bos_nnjm_id = nnjm_model->lookup_input_word("<src>");
    src_eos_nnjm_id = nnjm_model->lookup_input_word("</src>");
    tgt_bos_nnjm_id = nnjm_model->lookup_input_word("<tgt>");
    src_window_size = 5;
    tgt_window_size = 3;

    src_nnjm_ids.resize(src_window_size,src_bos_nnjm_id);
	stringstream ss(input_sen);
	string word;
    int i = 0;
    int beg = 0;
	while(ss>>word)
	{
        if (word == "EOS")
        {
            int span = i - beg - 1;                         //span=0表示句子包含1个词
            sen_spans.push_back(make_pair(beg,span));
            beg = i + 1;
            eos_indexes.push_back(i);
            src_wids.push_back(-1);
            src_nnjm_ids.push_back(-1);
        }
        else
        {
            src_wids.push_back(src_vocab->get_id(word));
            src_nnjm_ids.push_back(nnjm_model->lookup_input_word(word));
        }
        i++;
	}
    src_nnjm_ids.resize(src_nnjm_ids.size()+src_window_size,src_eos_nnjm_id);
	src_sen_len = src_wids.size();

    for (int i=0; i<src_sen_len; i++)
    {
        vector<int> cur_context(src_nnjm_ids.begin()+i,src_nnjm_ids.begin()+i+2*src_window_size+1);     //源端窗口长度为2*src_window_size+1
        bool flag = false;
        for (int j=src_window_size;j>=0;j--)
        {
            if (cur_context.at(j) == -1)
            {
                flag = true;
            }
            if (flag == true)
            {
                cur_context.at(j) = src_bos_nnjm_id;
            }
        }
        flag = false;
        for (int j=src_window_size;j<cur_context.size();j++)
        {
            if (cur_context.at(j) == -1)
            {
                flag = true;
            }
            if (flag == true)
            {
                cur_context.at(j) = src_eos_nnjm_id;
            }
        }
        src_context.push_back(cur_context);
    }

	span2validflag.resize(src_sen_len);
	sen_span_dict.resize(src_sen_len);
	span2cands.resize(src_sen_len);
	span2rules.resize(src_sen_len);
	for (size_t beg=0;beg<src_sen_len;beg++)
	{
		span2validflag.at(beg).resize(src_sen_len-beg,true);
		sen_span_dict.at(beg).resize(src_sen_len-beg,false);
		span2cands.at(beg).resize(src_sen_len-beg);
		span2rules.at(beg).resize(src_sen_len-beg);
	}

    for (auto sen_span : sen_spans)
    {
        sen_span_dict.at(sen_span.first).at(sen_span.second) = true;
    }

	fill_span2validflag();
	fill_span2cands_with_phrase_rules();
	fill_span2rules_with_hiero_rules();

    null_cand = new Cand;
    null_cand->rule_num = 0;
    null_cand->tgt_word_num = 0;
    null_cand->trans_probs.resize(PROB_NUM,0.0);
}

SentenceTranslator::~SentenceTranslator()
{
    delete null_cand;
	for (size_t i=0;i<span2cands.size();i++)
	{
		for(size_t j=0;j<span2cands.at(i).size();j++)
		{
			span2cands.at(i).at(j).free();
		}
	}
}

void SentenceTranslator::fill_span2validflag()
{
	for (size_t beg=0;beg<src_sen_len;beg++)
	{
		for (size_t span=0;beg+span<src_sen_len;span++)	//span=0对应跨度包含1个词的情况
		{
            int end = beg + span;
            for (int eos_idx : eos_indexes)
            {
                if (beg<=eos_idx && end>=eos_idx)
                {
                    span2validflag[beg][span] = false;
                    break;
                }
            }
		}
	}
}

/**************************************************************************************
 1. 函数功能: 根据规则表中匹配到的所有短语规则生成翻译候选, 并加入到span2cands中
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: a) 如果某个跨度没匹配到规则
              a.1) 如果该跨度包含1个单词, 则生成对应的OOV候选
              a.2) 如果该跨度包含多个单词, 则不作处理
              b) 如果某个跨度匹配到了规则, 则根据规则生成候选
************************************************************************************* */
void SentenceTranslator::fill_span2cands_with_phrase_rules()
{
	for (size_t beg=0;beg<src_sen_len;beg++)
	{
		vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(src_wids,beg);
		for (size_t span=0;span<matched_rules_for_prefixes.size();span++)	//span=0对应跨度包含1个词的情况
		{
            if (span2validflag[beg][span] == false)
                continue;
			if (matched_rules_for_prefixes.at(span) == NULL)
			{
				if (span == 0)
				{
					Cand* cand = new Cand;
					cand->tgt_wids.push_back(0 - src_wids.at(beg));
					cand->trans_probs.resize(PROB_NUM,0.0);
					cand->applied_rule.src_ids.push_back(src_wids.at(beg));
                    cand->applied_rule.span = make_pair(beg,span);
					cand->lm_prob = lm_model->cal_increased_lm_score(cand);
                    cand->aligned_src_idx.push_back(beg);
                    cand->span = make_pair(beg,span);
                    cand->nnjm_ngram_score.resize(1,0.0);
                    cand->nnjm_prob = cal_nnjm_score(cand);
					cand->score += feature_weight.rule_num*cand->rule_num + feature_weight.len*cand->tgt_word_num 
                                   + feature_weight.lm*cand->lm_prob + feature_weight.nnjm*cand->nnjm_prob;
					span2cands.at(beg).at(span).add(cand,para.BEAM_SIZE);
				}
				continue;
			}
			for (auto &tgt_rule : *matched_rules_for_prefixes.at(span))
			{
				Cand* cand = new Cand;
				cand->tgt_word_num = tgt_rule.word_num;
				cand->tgt_wids = tgt_rule.wids;
				cand->trans_probs = tgt_rule.probs;
				cand->score = tgt_rule.score;
				vector<int> src_ids(src_wids.begin()+beg,src_wids.begin()+beg+span+1);
				cand->applied_rule.src_ids = src_ids;
                cand->applied_rule.span = make_pair(beg,span);
				cand->applied_rule.tgt_rule = &tgt_rule;
				cand->lm_prob = lm_model->cal_increased_lm_score(cand);
                cand->aligned_src_idx = get_aligned_src_idx(beg,tgt_rule,NULL,NULL);
                cand->span = make_pair(beg,span);
                cand->nnjm_ngram_score.resize(cand->tgt_wids.size(),0.0);
                cand->nnjm_prob = cal_nnjm_score(cand);

				cand->score += feature_weight.rule_num*cand->rule_num + feature_weight.len*cand->tgt_word_num
                               + feature_weight.lm*cand->lm_prob + feature_weight.nnjm*cand->nnjm_prob;
				span2cands.at(beg).at(span).add(cand,para.BEAM_SIZE);
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 计算当前候选每个目标端单词对应的源端位置
 2. 入口参数: 当前候选对应的源端起始位置，规则内部每个目标端符号对应的规则源端位置
              当前候选的两个子候选
 3. 出口参数: 每个目标端单词对应的源端位置
 4. 算法简介: a) 对于有对齐的目标端单词，使用候选对应的源端起始位置加上该单词在规则内部
                 对应的源端位置，再加上非终结符导致的位置偏移
              b) 对于对空的目标端单词，使用临近的目标端单词对应的源端位置
              c) 对于目标端的非终结符，使用子候选的目标端单词到源端位置的映射
************************************************************************************* */
vector<int> SentenceTranslator::get_aligned_src_idx(int beg, TgtRule &tgt_rule, Cand* cand_x1, Cand* cand_x2)
{
    int nt1_idx = -1, nt2_idx = -1;
    int offset1 = 0, offset2 = 0;
    int nt_num = 1;
    for (int i=0; i<tgt_rule.tgt_to_src_idx.size(); i++)                  //记录非终结符在规则源端的位置以及每个非终结符*源端*包含的单词数
    {
        int tgt_wid = tgt_rule.wids.at(i);
        if (tgt_wid == tgt_nt_id && nt_num == 1)                          //第一个非终结符
        {
            nt1_idx = tgt_rule.tgt_to_src_idx.at(i);
            offset1 += cand_x1->span.second;
            nt_num++;
        }
        else if (tgt_wid == tgt_nt_id && nt_num == 2)                     //第二个非终结符
        {
            nt2_idx = tgt_rule.tgt_to_src_idx.at(i);
            offset2 += cand_x2->span.second;
        }
    }

    vector<int> tgt_to_src_idx_with_offset = tgt_rule.tgt_to_src_idx;
    for (int i=0; i<tgt_rule.tgt_to_src_idx.size(); i++)                  //处理偏置量
    {
        int src_idx = tgt_rule.tgt_to_src_idx.at(i);
        int tgt_wid = tgt_rule.wids.at(i);
        if (tgt_wid == tgt_nt_id || src_idx == -1)
            continue;
        if (nt1_idx != -1 && src_idx > nt1_idx)
        {
            tgt_to_src_idx_with_offset.at(i) += offset1;
        }
        if (nt2_idx != -1 && src_idx > nt2_idx)
        {
            tgt_to_src_idx_with_offset.at(i) += offset2;
        }
    }

    vector<int> aligned_src_idx;
    nt_num = 1;
    for (int i=0; i<tgt_to_src_idx_with_offset.size(); i++)               //将规则中的相对位置转换成句子中的绝对位置
    {
        int src_idx = tgt_to_src_idx_with_offset.at(i);
        int tgt_wid = tgt_rule.wids.at(i);
        if (tgt_wid == tgt_nt_id && nt_num == 1)                          //第一个非终结符
        {
            aligned_src_idx.insert(aligned_src_idx.end(),cand_x1->aligned_src_idx.begin(),cand_x1->aligned_src_idx.end());
            nt_num++;
        }
        else if (tgt_wid == tgt_nt_id && nt_num == 2)                     //第二个非终结符
        {
            aligned_src_idx.insert(aligned_src_idx.end(),cand_x2->aligned_src_idx.begin(),cand_x2->aligned_src_idx.end());
        }
        else if (src_idx == -1)
        {
            aligned_src_idx.push_back(-1);
        }
        else
        {
            aligned_src_idx.push_back(beg+src_idx);
        }
    }

    for (int i=0;i<aligned_src_idx.size();i++)      //处理对空的单词
    {
        if (aligned_src_idx.at(i) == -1)
        {
			for (int j=1; i+j < aligned_src_idx.size() || i-j >= 0; j++)
			{
				if( i+j < aligned_src_idx.size() && aligned_src_idx.at(i+j) != -1 )
				{
                    aligned_src_idx.at(i) = aligned_src_idx.at(i+j);
					break;
				}
				if( i-j >= 0 && aligned_src_idx.at(i-j) != -1 )
				{
                    aligned_src_idx.at(i) = aligned_src_idx.at(i-j);
					break;
				}
			}
        }
        assert(aligned_src_idx.at(i) != -1);
    }
    return aligned_src_idx;
}

/**************************************************************************************
 1. 函数功能: 计算当前候选的nnjm得分
 2. 入口参数: 当前候选
 3. 出口参数: 无
 4. 算法简介: 根据源端和目标端单词端位置获取计算每个nnjm ngram得分所需要的历史
************************************************************************************* */
double SentenceTranslator::cal_nnjm_score(Cand *cand)
{
    for (int tgt_idx=0;tgt_idx<cand->tgt_wids.size();tgt_idx++)
    {
        //if (cand->nnjm_ngram_score.at(tgt_idx) != 0.0)
            //continue;
        if (tgt_idx - tgt_window_size < 0 && sen_span_dict.at(cand->span.first).at(cand->span.second) == false)
            continue;

        vector<int> history = src_context.at(cand->aligned_src_idx.at(tgt_idx));
        for (int i = tgt_idx - tgt_window_size; i<tgt_idx; i++)
        {
            int nnjm_id = i<0 ? tgt_bos_nnjm_id : nnjm_model->lookup_input_word(get_tgt_word(cand->tgt_wids.at(i)));
            history.push_back(nnjm_id);
        }
        history.push_back(nnjm_model->lookup_output_word(get_tgt_word(cand->tgt_wids.at(tgt_idx))));
        auto it = nnjm_score_cache.find(history);
        if (it != nnjm_score_cache.end())
        {
            cand->nnjm_ngram_score.at(tgt_idx) = it->second;
        }
        else
        {
            double score = nnjm_model->lookup_ngram(history);
            cand->nnjm_ngram_score.at(tgt_idx) = score;
            nnjm_score_cache.insert(make_pair(history,score));
        }
    }
    return accumulate(cand->nnjm_ngram_score.begin(),cand->nnjm_ngram_score.end(),0.0);
}

string SentenceTranslator::get_tgt_word(int wid)
{
    if (wid>0)
        return tgt_vocab->get_word(wid);
    return src_vocab->get_word(0-wid);
}

/**************************************************************************************
 1. 函数功能: 找到每个跨度所有能用的hiero规则，并加入到span2rules中
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: 1) 找出当前句子所有可能的pattern，以及每个pattern对应的所有跨度
 			  2) 对每个pattern，检查规则表中是否存在可用的规则
 			  3) 根据每个可用的规则更新span2rules
************************************************************************************* */
void SentenceTranslator::fill_span2rules_with_hiero_rules()
{
	fill_span2rules_with_AX_XA_XAX_rule();                            //形如AX,XA和XAX的规则
	fill_span2rules_with_AXB_AXBX_XAXB_rule();                        //形如AXB,AXBX和XAXB的规则
	fill_span2rules_with_AXBXC_rule();                                //形如AXBXC的规则
	fill_span2rules_with_glue_rule();                                 //起始位置为句首，形如X1X2的规则
}

/**************************************************************************************
 1. 函数功能: 处理形如AX,XA,XAX的规则
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: 按照终结符序列的起始位置和长度遍历所有可能的pattern
			  p.s. beg_A+len_A为A的最后一个单词的位置
************************************************************************************* */
void SentenceTranslator::fill_span2rules_with_AX_XA_XAX_rule()
{
	for (int beg_A=0;beg_A<src_sen_len;beg_A++)
	{
		for (int len_A=0;beg_A+len_A<src_sen_len && len_A+1<=SPAN_LEN_MAX;len_A++)
		{
			vector<int> ids_A(src_wids.begin()+beg_A,src_wids.begin()+beg_A+len_A+1);
			//抽取形如XA的规则
			if (beg_A != 0)
			{
				vector<int> ids_XA;
				ids_XA.push_back(src_nt_id);
				ids_XA.insert(ids_XA.end(),ids_A.begin(),ids_A.end());
				vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(ids_XA,0);
				if (matched_rules_for_prefixes.size() == ids_XA.size() && matched_rules_for_prefixes.back() != NULL)         //找到了可用的规则
				{
					for (int len_X=0;len_X<beg_A && len_X+len_A+2<=SPAN_LEN_MAX;len_X++)
					{
						int beg_X = beg_A - len_X - 1;
						pair<int,int> span = make_pair(beg_X,len_X+len_A+1);
						pair<int,int> span_src_x1 = make_pair(beg_X,len_X);
						pair<int,int> span_src_x2 = make_pair(-1,-1);
						fill_span2rules_with_matched_rules(*matched_rules_for_prefixes.back(),ids_XA,span,span_src_x1,span_src_x2);
					}
				}
			}
			//抽取形如AX的规则
			if (beg_A+len_A != src_sen_len - 1)
			{
				vector<int> ids_AX;
				ids_AX = ids_A;
				ids_AX.push_back(src_nt_id);
				vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(ids_AX,0);
				if (matched_rules_for_prefixes.size() == ids_AX.size() && matched_rules_for_prefixes.back() != NULL)         //找到了可用的规则
				{
					for (int len_X=0;beg_A+len_A+1+len_X<src_sen_len && len_A+len_X+2<=SPAN_LEN_MAX;len_X++)
					{
						int beg_X = beg_A + len_A + 1;
						pair<int,int> span = make_pair(beg_A,len_A+len_X+1);
						pair<int,int> span_src_x1 = make_pair(beg_X,len_X);
						pair<int,int> span_src_x2 = make_pair(-1,-1);
						fill_span2rules_with_matched_rules(*matched_rules_for_prefixes.back(),ids_AX,span,span_src_x1,span_src_x2);
					}
				}
			}
			//抽取形如XAX的规则
			if (beg_A != 0 && beg_A+len_A != src_sen_len - 1)
			{
				vector<int> ids_XAX;
				ids_XAX.push_back(src_nt_id);
				ids_XAX.insert(ids_XAX.end(),ids_A.begin(),ids_A.end());
				ids_XAX.push_back(src_nt_id);
				vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(ids_XAX,0);
				if (matched_rules_for_prefixes.size() == ids_XAX.size() && matched_rules_for_prefixes.back() != NULL)         //找到了可用的规则
				{
					for (int len_X1=0;len_X1<beg_A && len_X1+len_A+2<=SPAN_LEN_MAX-1;len_X1++)
					{
						for (int len_X2=0;beg_A+len_A+1+len_X2<src_sen_len && len_X1+len_A+len_X2<=SPAN_LEN_MAX;len_X2++)
						{
							int beg_X1 = beg_A - len_X1 - 1;
							int beg_X2 = beg_A + len_A + 1;
							pair<int,int> span = make_pair(beg_X1,len_X1+len_A+len_X2+2);
							pair<int,int> span_src_x1 = make_pair(beg_X1,len_X1);
							pair<int,int> span_src_x2 = make_pair(beg_X2,len_X2);
							fill_span2rules_with_matched_rules(*matched_rules_for_prefixes.back(),ids_XAX,span,span_src_x1,span_src_x2);
						}
					}
				}
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 处理形如AXB,AXBX,XAXB的规则
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: 按照终结符序列的起始位置和长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::fill_span2rules_with_AXB_AXBX_XAXB_rule()
{
	for (int beg_AXB=0;beg_AXB<src_sen_len;beg_AXB++)
	{
		for (int len_AXB=0;beg_AXB+len_AXB<src_sen_len && len_AXB<=SPAN_LEN_MAX;len_AXB++)
		{
			for (int beg_X=beg_AXB+1;beg_X<beg_AXB+len_AXB;beg_X++)
			{
				for (int len_X=0;beg_X+len_X<beg_AXB+len_AXB;len_X++)
				{
					vector<int> ids_AXB(src_wids.begin()+beg_AXB,src_wids.begin()+beg_X);
					ids_AXB.push_back(src_nt_id);
					ids_AXB.insert(ids_AXB.end(),src_wids.begin()+beg_X+len_X+1,src_wids.begin()+beg_AXB+len_AXB+1);
					//抽取形如XAXB的pattern
					if (beg_AXB != 0)
					{
						vector<int> ids_XAXB;
						ids_XAXB.push_back(src_nt_id);
						ids_XAXB.insert(ids_XAXB.end(),ids_AXB.begin(),ids_AXB.end());
						vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(ids_XAXB,0);
						if (matched_rules_for_prefixes.size() == ids_XAXB.size() && matched_rules_for_prefixes.back() != NULL)         //找到了可用的规则
						{
							for (int len_X1=0;len_X1<beg_AXB && len_X1+len_AXB+2<=SPAN_LEN_MAX;len_X1++)
							{
								int beg_X1 = beg_AXB - len_X1 - 1;
								pair<int,int> span = make_pair(beg_X1,len_X1+len_AXB+1);
								pair<int,int> span_src_x1 = make_pair(beg_X1,len_X1);
								pair<int,int> span_src_x2 = make_pair(beg_X,len_X);
								fill_span2rules_with_matched_rules(*matched_rules_for_prefixes.back(),ids_XAXB,span,span_src_x1,span_src_x2);
							}
						}
					}
					//抽取形如AXBX的pattern
					if (beg_AXB+len_AXB != src_sen_len - 1)
					{
						vector<int> ids_AXBX;
						ids_AXBX = ids_AXB;
						ids_AXBX.push_back(src_nt_id);
						vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(ids_AXBX,0);
						if (matched_rules_for_prefixes.size() == ids_AXBX.size() && matched_rules_for_prefixes.back() != NULL)         //找到了可用的规则
						{
							for (int len_X2=0;beg_AXB+len_AXB+1+len_X2<src_sen_len && len_AXB+len_X2+2<=SPAN_LEN_MAX;len_X2++)
							{
								int beg_X2 = beg_AXB + len_AXB + 1;
								pair<int,int> span = make_pair(beg_AXB,len_AXB+len_X2+1);
								pair<int,int> span_src_x1 = make_pair(beg_X,len_X);
								pair<int,int> span_src_x2 = make_pair(beg_X2,len_X2);
								fill_span2rules_with_matched_rules(*matched_rules_for_prefixes.back(),ids_AXBX,span,span_src_x1,span_src_x2);
							}
						}
					}
					//抽取形如AXB的pattern
					vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(ids_AXB,0);
					if (matched_rules_for_prefixes.size() == ids_AXB.size() && matched_rules_for_prefixes.back() != NULL)         //找到了可用的规则
					{
						pair<int,int> span = make_pair(beg_AXB,len_AXB);
						pair<int,int> span_src_x1 = make_pair(beg_X,len_X);
						pair<int,int> span_src_x2 = make_pair(-1,-1);
						fill_span2rules_with_matched_rules(*matched_rules_for_prefixes.back(),ids_AXB,span,span_src_x1,span_src_x2);
					}
				}
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 处理形如AXBXC的规则
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: 按照终结符序列的起始位置和长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::fill_span2rules_with_AXBXC_rule()
{
	for (int beg_AXBXC=0;beg_AXBXC<src_sen_len;beg_AXBXC++)
	{
		for (int len_AXBXC=4;beg_AXBXC+len_AXBXC<src_sen_len && len_AXBXC<=SPAN_LEN_MAX;len_AXBXC++)
		{
			for (int beg_XBX=beg_AXBXC+1;beg_XBX+2<beg_AXBXC+len_AXBXC;beg_XBX++)
			{
				for (int len_XBX=0;beg_XBX+len_XBX<beg_AXBXC+len_AXBXC;len_XBX++)
				{
					for (int beg_B=beg_XBX+1;beg_B<beg_XBX+len_XBX;beg_B++)
					{
						for (int len_B=0;len_B<len_XBX-(beg_B-beg_XBX);len_B++)
						{
							//抽取形如AXBXC的pattern
							vector<int> ids_AXBXC(src_wids.begin()+beg_AXBXC,src_wids.begin()+beg_XBX);
							ids_AXBXC.push_back(src_nt_id);
							ids_AXBXC.insert(ids_AXBXC.end(),src_wids.begin()+beg_B,src_wids.begin()+beg_B+len_B+1);
							ids_AXBXC.push_back(src_nt_id);
							ids_AXBXC.insert(ids_AXBXC.end(),src_wids.begin()+beg_XBX+len_XBX+1,src_wids.begin()+beg_AXBXC+len_AXBXC+1);
							vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(ids_AXBXC,0);
							if (matched_rules_for_prefixes.size() == ids_AXBXC.size() && matched_rules_for_prefixes.back() != NULL)         //找到了可用的规则
							{
								pair<int,int> span = make_pair(beg_AXBXC,len_AXBXC);
								pair<int,int> span_src_x1 = make_pair(beg_XBX,beg_B-beg_XBX-1);
								pair<int,int> span_src_x2 = make_pair(beg_B+len_B+1,len_XBX-len_B-(beg_B-beg_XBX-1)-2);
								fill_span2rules_with_matched_rules(*matched_rules_for_prefixes.back(),ids_AXBXC,span,span_src_x1,span_src_x2);
							}
						}
					}
				}
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 处理glue规则
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: 按照第一个非终结符的长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::fill_span2rules_with_glue_rule()
{
	vector<int> ids_X1X2 = {src_nt_id,src_nt_id};
	vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(ids_X1X2,0);

    for (auto &sen_span : sen_spans)
    {
        int sen_beg = sen_span.first;
        int sen_len = sen_span.second;
        for (int len_X1X2=1;len_X1X2<=sen_len;len_X1X2++)               //glue pattern的跨度不受规则最大跨度RULE_LEN_MAX的限制，可以延伸到句尾
        {
            for (int len_X1=0;len_X1<len_X1X2;len_X1++)
            {
                Rule rule;
                rule.src_ids = ids_X1X2;
                rule.tgt_rule = &((*matched_rules_for_prefixes.back()).at(0));
                rule.tgt_rule_rank = 0;
                rule.span_x1 = make_pair(sen_beg,len_X1);
                rule.span_x2 = make_pair(sen_beg+len_X1+1,len_X1X2-len_X1-1);
                span2rules.at(sen_beg).at(len_X1X2).push_back(rule);
            }
        }
    }
}

/**************************************************************************************
 1. 函数功能: 对给定的pattern以及该pattern对应的span，将匹配到的规则加入span2rules中
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: 略
************************************************************************************* */
void SentenceTranslator::fill_span2rules_with_matched_rules(vector<TgtRule> &matched_rules,vector<int> &src_ids,pair<int,int> span,pair<int,int> span_src_x1,pair<int,int> span_src_x2)
{
    if (span2validflag[span.first][span.second] == false)
        return;
	for (int i=0;i<matched_rules.size();i++)
	{
		Rule rule;
        rule.span = span;
		rule.src_ids = src_ids;
		rule.tgt_rule = &matched_rules.at(i);
		rule.tgt_rule_rank = i;
		if (matched_rules.at(i).rule_type == 3)
		{
			rule.span_x1 = span_src_x2;
			rule.span_x2 = span_src_x1;
		}
		else
		{
			rule.span_x1 = span_src_x1;
			rule.span_x2 = span_src_x2;
		}
		span2rules.at(span.first).at(span.second).push_back(rule);
	}
}

string SentenceTranslator::words_to_str(vector<int> wids, int drop_oov)
{
		string output = "";
		for (const auto &wid : wids)
		{
			if (wid >= 0)
			{
				output += tgt_vocab->get_word(wid) + " ";
			}
			else if (drop_oov == 0)
			{
				output += src_vocab->get_word(0-wid) + " ";
			}
		}
		TrimLine(output);
		return output;
}

vector<vector<TuneInfo> > SentenceTranslator::get_tune_info(size_t sen_id)
{
	vector<vector<TuneInfo> > nbest_tune_info_list;
    for (auto &sen_span : sen_spans)
    {
        vector<TuneInfo> nbest_tune_info;
        CandBeam &candbeam = span2cands.at(sen_span.first).at(sen_span.second);
        for (size_t i=0;i< (candbeam.size()<para.NBEST_NUM?candbeam.size():para.NBEST_NUM);i++)
        {
            TuneInfo tune_info;
            tune_info.sen_id = sen_id;
            tune_info.translation = words_to_str(candbeam.at(i)->tgt_wids,0);
            for (size_t j=0;j<PROB_NUM;j++)
            {
                tune_info.feature_values.push_back(candbeam.at(i)->trans_probs.at(j));
            }
            tune_info.feature_values.push_back(candbeam.at(i)->lm_prob);
            tune_info.feature_values.push_back(candbeam.at(i)->tgt_word_num);
            tune_info.feature_values.push_back(candbeam.at(i)->rule_num);
            tune_info.feature_values.push_back(candbeam.at(i)->glue_num);
            tune_info.feature_values.push_back(candbeam.at(i)->nnjm_prob);
            tune_info.total_score = candbeam.at(i)->score;
            nbest_tune_info.push_back(tune_info);
        }
        nbest_tune_info_list.push_back(nbest_tune_info);
    }
	return nbest_tune_info_list;
}

vector<vector<string> > SentenceTranslator::get_applied_rules(size_t sen_id)
{
    vector<vector<string> > applied_rules_list;
    for (auto &sen_span : sen_spans)
    {
        vector<string> applied_rules;
        Cand *best_cand = span2cands.at(sen_span.first).at(sen_span.second).top();
        dump_rules(applied_rules,best_cand);
        applied_rules.push_back(" ||||| ");
        string src_sen;
        for (auto wid : src_wids)
        {
            src_sen += src_vocab->get_word(wid)+" ";
        }
        applied_rules.push_back(src_sen);
        applied_rules_list.push_back(applied_rules);
    }
	return applied_rules_list;
}

/**************************************************************************************
 1. 函数功能: 获取当前候选所使用的规则
 2. 入口参数: 当前候选的指针
 3. 出口参数: 用于记录规则的applied_rules
 4. 算法简介: 通过递归的方式回溯, 如果当前候选没有子候选, 则找到了一条规则, 否则获取
 			  子候选所使用的规则
************************************************************************************* */
void SentenceTranslator::dump_rules(vector<string> &applied_rules, Cand *cand)
{
	applied_rules.push_back(" ");
	if (cand->child_x1 != NULL)
	{
		applied_rules.push_back(" ( ");
	}
	string rule;
	int nt_num = 0;
	vector<string> src_nts = {"X1_","X2_"};
	vector<string> tgt_nts = {"X1_","X2_"};
	vector<string> src_spans = 
	{"(_"+to_string(cand->applied_rule.span_x1.first)+"-"+to_string(cand->applied_rule.span_x1.first+cand->applied_rule.span_x1.second)+"_)_",
	"(_"+to_string(cand->applied_rule.span_x2.first)+"-"+to_string(cand->applied_rule.span_x2.first+cand->applied_rule.span_x2.second)+"_)_"};
	vector<Cand*> children = {cand->child_x1,cand->child_x2};
	if (cand->applied_rule.tgt_rule != NULL && cand->applied_rule.tgt_rule->rule_type == 3)
	{
		reverse(src_spans.begin(),src_spans.end());
		reverse(tgt_nts.begin(),tgt_nts.end());
		reverse(children.begin(),children.end());
	}
	for (auto src_wid : cand->applied_rule.src_ids)
	{
		if (src_wid == src_nt_id)
		{
			rule += src_nts[nt_num];
			//rule += src_spans[nt_num];
			nt_num++;
		}
		else
		{
			rule += src_vocab->get_word(src_wid)+"_";
		}
	}
	rule += "|||_";
	if (cand->applied_rule.tgt_rule == NULL)
	{
		rule += "NULL_";
	}
	else
	{
		nt_num = 0;
		for (auto tgt_wid : cand->applied_rule.tgt_rule->wids)
		{
			if (tgt_wid == tgt_nt_id)
			{
				rule += tgt_nts[nt_num];
				nt_num++;
			}
			else
			{
				rule += tgt_vocab->get_word(tgt_wid)+"_";
			}
		}
	}
	rule.erase(rule.end()-1);
	applied_rules.push_back(rule);
	if (children[0] != NULL)
	{
		dump_rules(applied_rules,children[0]);
	}
	if (children[1] != NULL)
	{
		dump_rules(applied_rules,children[1]);
	}
	if (cand->child_x1 != NULL)
	{
		applied_rules.push_back(" ) ");
	}
}

vector<string> SentenceTranslator::translate_sentence()
{
	for(size_t beg=0;beg<src_sen_len;beg++)
	{
		span2cands.at(beg).at(0).sort();		               //对列表中的候选进行排序
	}
	for (size_t span=1;span<src_sen_len;span++)
	{
//#pragma omp parallel for num_threads(para.SPAN_THREAD_NUM)
		for(size_t beg=0;beg<src_sen_len-span;beg++)
		{
			generate_kbest_for_span(beg,span);
			span2cands.at(beg).at(span).sort();
		}
	}
    vector<string> output_sens;
    for (auto &sen_span : sen_spans)
    {
        output_sens.push_back(words_to_str(span2cands.at(sen_span.first).at(sen_span.second).top()->tgt_wids,para.DROP_OOV));
    }
    return output_sens;
}

/**************************************************************************************
 1. 函数功能: 为每个跨度生成kbest候选
 2. 入口参数: 跨度的起始位置以及跨度的长度(实际为长度减1)
 3. 出口参数: 无
 4. 算法简介: 见注释
************************************************************************************* */
void SentenceTranslator::generate_kbest_for_span(const size_t beg,const size_t span)
{
    if (span2validflag[beg][span] == false)
        return;
	Candpq candpq_merge;			    //优先级队列,用来临时存储通过合并得到的候选
	set<vector<int> > duplicate_set;	//用来记录候选是否已经被加入candpq_merge中

	//对于当前跨度匹配到的每一条规则,取出非终结符对应的跨度中的最好候选,将合并得到的候选加入candpq_merge
	for(auto &rule : span2rules.at(beg).at(span))
	{
		generate_cand_with_rule_and_add_to_pq(rule,0,0,candpq_merge,duplicate_set);
	}

	//立方体剪枝,每次从candpq_merge中取出最好的候选加入span2cands中,并将该候选的邻居加入candpq_merge中
	int added_cand_num = 0;
	while (added_cand_num<para.CUBE_SIZE)
	{
		if (candpq_merge.empty()==true)
			break;
		Cand* best_cand = candpq_merge.top();
		candpq_merge.pop();
        bool flag = false;
        for (auto &sen_span : sen_spans)
        {
            if (beg == sen_span.first && span == sen_span.second)
            {
                flag = true;
                break;
            }
        }
		if (flag == true)
		{
			double increased_lm_prob = lm_model->cal_final_increased_lm_score(best_cand);
			best_cand->lm_prob += increased_lm_prob;
			best_cand->score += feature_weight.lm*increased_lm_prob;
		}
		
        add_neighbours_to_pq(best_cand,candpq_merge,duplicate_set);
		span2cands.at(beg).at(span).add(best_cand,para.BEAM_SIZE);
		added_cand_num++;
	}

	while(!candpq_merge.empty())
	{
		delete candpq_merge.top();
		candpq_merge.pop();
	}
}

/**************************************************************************************
 1. 函数功能: 合并两个子候选并将生成的候选加入candpq_merge中
 2. 入口参数: 两个子候选,两个子候选的排名
 3. 出口参数: 更新后的candpq_merge
 4. 算法简介: 顺序以及逆序合并两个子候选
************************************************************************************* */
void SentenceTranslator::generate_cand_with_rule_and_add_to_pq(Rule &rule,int rank_x1,int rank_x2,Candpq &candpq_merge,set<vector<int> > &duplicate_set)
{
    //key包含两个变量在源端的span（用来检查规则源端是否相同），规则目标端在源端相同的所有目标端的排名（检查规则目标端是否相同）
    //以及子候选在两个变量中的排名（检查子候选是否相同）
    vector<int> key = {rule.span_x1.first,rule.span_x1.second,rule.span_x2.first,rule.span_x2.second,rule.tgt_rule_rank,rank_x1,rank_x2};
    if (duplicate_set.insert(key).second == false)
        return;

    if (span2cands.at(rule.span_x1.first).at(rule.span_x1.second).size() <= rank_x1)
        return;
    if (rule.tgt_rule->rule_type >=2 && span2cands.at(rule.span_x2.first).at(rule.span_x2.second).size() <= rank_x2)
        return;

    Cand *cand_x1 = span2cands.at(rule.span_x1.first).at(rule.span_x1.second).at(rank_x1);
    Cand *cand_x2 = rule.tgt_rule->rule_type >= 2 ? span2cands.at(rule.span_x2.first).at(rule.span_x2.second).at(rank_x2) : null_cand;
    Cand *cand = new Cand;
    update_cand_members(cand,rule,rank_x1,rank_x2,cand_x1,cand_x2);
    candpq_merge.push(cand);
}

void SentenceTranslator::update_cand_members(Cand* cand, Rule &rule, int rank_x1, int rank_x2, Cand* cand_x1, Cand* cand_x2)
{
    cand->span = rule.span;
    cand->applied_rule = rule;
    int glue_num = rule.tgt_rule->rule_type == 4 ? 1 : 0;
    cand->rule_num = cand_x1->rule_num + cand_x2->rule_num + 1;
    cand->glue_num = cand_x1->glue_num + cand_x2->glue_num + glue_num;
    cand->rank_x1 = rank_x1;
    cand->rank_x2 = rule.tgt_rule->rule_type >= 2 ? rank_x2 : -1;
    cand->child_x1 = cand_x1;
    cand->child_x2 = rule.tgt_rule->rule_type >= 2 ? cand_x2 : NULL;
    cand->tgt_word_num = cand_x1->tgt_word_num + cand_x2->tgt_word_num + rule.tgt_rule->word_num;

    cand->aligned_src_idx = get_aligned_src_idx(cand->span.first,*(rule.tgt_rule),cand_x1,cand_x2);

    int nt_num = 1; 							//表示第几个非终结符
    for (auto tgt_wid : rule.tgt_rule->wids)
    {
        if (tgt_wid == tgt_nt_id)
        {
            Cand* sub_cand = nt_num == 1 ? cand_x1 : cand_x2;
            cand->tgt_wids.insert(cand->tgt_wids.end(),sub_cand->tgt_wids.begin(),sub_cand->tgt_wids.end());
            cand->nnjm_ngram_score.insert(cand->nnjm_ngram_score.end(),sub_cand->nnjm_ngram_score.begin(),sub_cand->nnjm_ngram_score.end());
            nt_num++;
        }
        else
        {
            cand->tgt_wids.push_back(tgt_wid);
            cand->nnjm_ngram_score.push_back(0.0);
        }
    }
    for (size_t i=0;i<PROB_NUM;i++)
    {
        cand->trans_probs.push_back(cand_x1->trans_probs.at(i) + cand_x2->trans_probs.at(i) + rule.tgt_rule->probs.at(i));
    }
    cand->nnjm_prob = cal_nnjm_score(cand);
    double increased_nnjm_prob = cand->nnjm_prob - cand_x1->nnjm_prob - cand_x2->nnjm_prob;
    double increased_lm_prob = lm_model->cal_increased_lm_score(cand);
    cand->lm_prob = cand_x1->lm_prob + cand_x2->lm_prob + increased_lm_prob;
    cand->score = cand_x1->score + cand_x2->score + rule.tgt_rule->score + feature_weight.lm*increased_lm_prob
        + feature_weight.rule_num*1 + feature_weight.glue*glue_num + feature_weight.len*rule.tgt_rule->word_num
        + feature_weight.nnjm*increased_nnjm_prob;
}

/**************************************************************************************
 1. 函数功能: 将当前候选的邻居加入candpq_merge中
 2. 入口参数: 当前候选
 3. 出口参数: 更新后的candpq_merge
 4. 算法简介: a) 取比当前候选左子候选差一名的候选与当前候选的右子候选合并
              b) 取比当前候选右子候选差一名的候选与当前候选的左子候选合并
************************************************************************************* */
void SentenceTranslator::add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq_merge,set<vector<int> > &duplicate_set)
{
	if (cur_cand->rank_x2 != -1)                                                //如果生成当前候选的规则包括两个非终结符
	{
		int rank_x1 = cur_cand->rank_x1 + 1;
		int rank_x2 = cur_cand->rank_x2;
		generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge,duplicate_set);

		rank_x1 = cur_cand->rank_x1;
		rank_x2 = cur_cand->rank_x2 + 1;
		generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge,duplicate_set);
	}
	else 																		//如果生成当前候选的规则包括一个非终结符
	{
		int rank_x1 = cur_cand->rank_x1 + 1;
		int rank_x2 = cur_cand->rank_x2;
		generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge,duplicate_set);
	}
}

void SentenceTranslator::show_cand(Cand* cand)
{
    for (int i=cand->span.first;i<=cand->span.first+cand->span.second;i++)
        cout<<src_vocab->get_word(src_wids.at(i))<<' ';
    cout<<"||| ";
    for (int i=0; i<cand->tgt_wids.size(); i++)
    {
        cout<<get_tgt_word(cand->tgt_wids.at(i))<<'/';
        cout<<src_vocab->get_word(src_wids.at(cand->aligned_src_idx.at(i)))<<'/';
        cout<<cand->nnjm_ngram_score.at(i)<<' ';
    }
    cout<<endl;
}

void SentenceTranslator::show_rule(Rule &rule)
{
    if (rule.tgt_rule == NULL)
        return;
    for (int e : rule.src_ids)
        cout<<src_vocab->get_word(e)<<' ';
    cout<<"||| ";
    for (int i=0; i<rule.tgt_rule->wids.size(); i++)
    {
        cout<<get_tgt_word(rule.tgt_rule->wids.at(i))<<'/';
        cout<<rule.tgt_rule->tgt_to_src_idx.at(i)<<' ';
    }
    cout<<endl;
}
