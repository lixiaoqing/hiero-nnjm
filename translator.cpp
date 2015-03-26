#include "translator.h"

SentenceTranslator::SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen)
{
	src_vocab = i_models.src_vocab;
	tgt_vocab = i_models.tgt_vocab;
	ruletable = i_models.ruletable;
	lm_model = i_models.lm_model;
	para = i_para;
	feature_weight = i_weight;

	stringstream ss(input_sen);
	string word;
	while(ss>>word)
	{
		src_wids.push_back(src_vocab->get_id(word));
	}

	src_sen_len = src_wids.size();
	candbeam_matrix.resize(src_sen_len);
	for (size_t beg=0;beg<src_sen_len;beg++)
	{
		candbeam_matrix.at(beg).resize(src_sen_len-beg);
	}

	fill_matrix_with_matched_rules();
}

SentenceTranslator::~SentenceTranslator()
{
	for (size_t i=0;i<candbeam_matrix.size();i++)
	{
		for(size_t j=0;j<candbeam_matrix.at(i).size();j++)
		{
			candbeam_matrix.at(i).at(j).free();
		}
	}
}

/**************************************************************************************
 1. 函数功能: 根据规则表中匹配到的所有短语规则生成翻译候选, 并加入到candbeam_matrix中
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: a) 如果某个跨度没匹配到规则
              a.1) 如果该跨度包含1个单词, 则生成对应的OOV候选
              a.2) 如果该跨度包含多个单词, 则不作处理
              b) 如果某个跨度匹配到了规则, 则根据规则生成候选
************************************************************************************* */
void SentenceTranslator::fill_matrix_with_matched_rules()
{
	for (size_t beg=0;beg<src_sen_len;beg++)
	{
		vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(src_wids,beg);
		for (size_t span=0;span<matched_rules_for_prefixes.size();span++)	//span=0对应跨度包含1个词的情况
		{
			if (matched_rules_for_prefixes.at(span) == NULL)
			{
				if (span == 0)
				{
					Cand* cand = new Cand;
					cand->tgt_wids.push_back(tgt_vocab->get_id("NULL"));
					cand->trans_probs.resize(PROB_NUM,LogP_PseudoZero);
					for (size_t i=0;i<PROB_NUM;i++)
					{
						cand->score += feature_weight.trans.at(i)*cand->trans_probs.at(i);
					}
					cand->lm_prob = lm_model->cal_increased_lm_score(cand);
					cand->score += feature_weight.rule_num*cand->rule_num 
						       + feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
					candbeam_matrix.at(beg).at(span).add(cand);
				}
				continue;
			}
			for (const auto &tgt_rule : *matched_rules_for_prefixes.at(span))
			{
				Cand* cand = new Cand;
				cand->tgt_word_num = tgt_rule.word_num;
				cand->tgt_wids = tgt_rule.wids;
				cand->trans_probs = tgt_rule.probs;
				cand->score = tgt_rule.score;
				cand->lm_prob = lm_model->cal_increased_lm_score(cand);
				cand->score += feature_weight.rule_num*cand->rule_num 
					       + feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
				candbeam_matrix.at(beg).at(span).add(cand);
			}
		}
	}
}

string SentenceTranslator::words_to_str(vector<int> wids, bool drop_unk)
{
		string output = "";
		for (const auto &wid : wids)
		{
			string word = tgt_vocab->get_word(wid);
			if (word != "NULL" || drop_unk == false)
			{
				output += word + " ";
			}
		}
		TrimLine(output);
		return output;
}

vector<TuneInfo> SentenceTranslator::get_tune_info(size_t sen_id)
{
	vector<TuneInfo> nbest_tune_info;
	CandBeam &candbeam = candbeam_matrix.at(0).at(src_sen_len-1);
	for (size_t i=0;i< (candbeam.size()<para.NBEST_NUM?candbeam.size():para.NBEST_NUM);i++)
	{
		TuneInfo tune_info;
		tune_info.sen_id = sen_id;
		tune_info.translation = words_to_str(candbeam.at(i)->tgt_wids,false);
		for (size_t j=0;j<PROB_NUM;j++)
		{
			tune_info.feature_values.push_back(candbeam.at(i)->trans_probs.at(j));
		}
		tune_info.feature_values.push_back(candbeam.at(i)->lm_prob);
		tune_info.feature_values.push_back(candbeam.at(i)->tgt_word_num);
		tune_info.feature_values.push_back(candbeam.at(i)->rule_num);
		tune_info.feature_values.push_back(candbeam.at(i)->glue_num);
		tune_info.total_score = candbeam.at(i)->score;
		nbest_tune_info.push_back(tune_info);
	}
	return nbest_tune_info;
}

vector<string> SentenceTranslator::get_applied_rules(size_t sen_id)
{
	vector<string> applied_rules;
	Cand *best_cand = candbeam_matrix.at(0).at(src_sen_len-1).top();
	dump_rules(applied_rules,best_cand);
	return applied_rules;
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
	string rule;
	for (auto src_wid : cand->applied_rule.src_rule)
	{
		rule += src_vocab->get_word(src_wid)+" ";
	}
	rule += "||| ";
	for (auto tgt_wid : cand->applied_rule.tgt_rule)
	{
		rule += tgt_vocab->get_word(tgt_wid)+" ";
	}
	TrimLine(rule);
	applied_rules.push_back(rule);
	dump_rules(applied_rules,cand->child_x1);
	if (cand->child_x2 != NULL)
	{
		dump_rules(applied_rules,cand->child_x2);
	}
}

string SentenceTranslator::translate_sentence()
{
	if (src_sen_len == 0)
		return "";
	for(size_t beg=0;beg<src_sen_len;beg++)
	{
		candbeam_matrix.at(beg).at(0).sort();		//对列表中的候选进行排序
	}
	for (size_t span=1;span<src_sen_len;span++)
	{
#pragma omp parallel for num_threads(para.SPAN_THREAD_NUM)
		for(size_t beg=0;beg<src_sen_len-span;beg++)
		{
			generate_kbest_for_span(beg,span);
			candbeam_matrix.at(beg).at(span).sort();
		}
	}
	return words_to_str(candbeam_matrix.at(0).at(src_sen_len-1).top()->tgt_wids,true);
}

/**************************************************************************************
 1. 函数功能: 为每个跨度生成kbest候选
 2. 入口参数: 跨度的起始位置以及跨度的长度(实际为长度减1)
 3. 出口参数: 无
 4. 算法简介: 见注释
************************************************************************************* */
void SentenceTranslator::generate_kbest_for_span(const size_t beg,const size_t span)
{
	Candpq candpq_merge;			//优先级队列,用来临时存储通过合并得到的候选

	//生成能与当前跨度对应的字符串匹配的所有pattern，一一拿到规则表中匹配，找出能用的规则
	vector<Pattern> possible_patterns;
	get_patterns_with_one_terminal(beg,span,possible_patterns);
	get_patterns_with_two_terminals(beg,span,possible_patterns);
	get_patterns_for_glue_rule(beg,span,possible_patterns);

	vector<Rule> applicable_rules;
	//对于当前跨度匹配到的每一条规则,取出非终结符对应的跨度中的最好候选,将合并得到的候选加入candpq_merge
	for(auto &rule : applicable_rules)
	{
		generate_cand_with_rule_and_add_to_pq(rule,1,1,candpq_merge);
	}

	//立方体剪枝,每次从candpq_merge中取出最好的候选加入candbeam_matrix中,并将该候选的邻居加入candpq_merge中
	int added_cand_num = 0;
	while (added_cand_num<para.BEAM_SIZE)
	{
		if (candpq_merge.empty()==true)
			break;
		Cand* best_cand = candpq_merge.top();
		candpq_merge.pop();
		if (span == src_sen_len-1)
		{
			double increased_lm_prob = lm_model->cal_final_increased_lm_score(best_cand);
			best_cand->lm_prob += increased_lm_prob;
			best_cand->score += feature_weight.lm*increased_lm_prob;
		}
		
		add_neighbours_to_pq(best_cand,candpq_merge);
		bool flag = candbeam_matrix.at(beg).at(span).add(best_cand);
		if (flag == false)					//如果被丢弃
		{
			delete best_cand;
		}
		else
		{
			added_cand_num++;
		}
	}
	while(!candpq_merge.empty())
	{
		delete candpq_merge.top();
		candpq_merge.pop();
	}
}

/**************************************************************************************
 1. 函数功能: 获取当前跨度能匹配的所有包含一个非终结符的pattern
 2. 入口参数: 当前跨度的起始位置和长度
 3. 出口参数: 能匹配的pattern
 4. 算法简介: 按照非终结符的起始位置和长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::get_patterns_with_one_terminal(const size_t beg,const size_t span,vector<Pattern> &possible_patterns)
{
	if (span == 0)                                          //当前span只包含一个单词
		return;
	for (int nt_beg=beg;nt_beg<beg+span+1;nt_beg++)
	{
		for (int nt_span=0;nt_span<beg+span+1-nt_beg && nt_span<span;nt_span++)
		{
			vector<int> src_ids;
			src_ids.insert(src_ids.end(),src_wids.begin()+beg,src_wids.begin()+nt_beg);
			src_ids.push_back(src_vocab->get_id("[X][X]"));
			src_ids.insert(src_ids.end(),src_wids.begin()+nt_beg+nt_span+1,src_wids.begin()+beg+span+1);
			Pattern pattern;
			pattern.src_ids = src_ids;
			pattern.span_src_x1 = make_pair(nt_beg,nt_span);
			pattern.span_src_x2 = make_pair(-1,-1);
			possible_patterns.push_back(pattern);
		}
	}
}

/**************************************************************************************
 1. 函数功能: 获取当前跨度能匹配的所有包含两个非终结符的pattern
 2. 入口参数: 当前跨度的起始位置和长度
 3. 出口参数: 能匹配的pattern
 4. 算法简介: 按照非终结符的起始位置和长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::get_patterns_with_two_terminals(const size_t beg,const size_t span,vector<Pattern> &possible_patterns)
{
	if (span <= 1)                                          //当前span包含不到三个单词
		return;
	for (int nt1_beg=beg;nt1_beg<beg+span;nt1_beg++)
	{
		for (int nt1_span=0;nt1_span<beg+span-nt1_beg;nt1_span++)
		{
			for (int nt2_beg=nt1_beg+nt1_span+2;nt2_beg<beg+span+1;nt2_beg++)
			{
				for (int nt2_span=0;nt2_span<beg+span+1-nt2_beg;nt2_span++)
				{
					vector<int> src_ids;
					src_ids.insert(src_ids.end(),src_wids.begin()+beg,src_wids.begin()+beg+nt1_beg);
					src_ids.push_back(src_vocab->get_id("[X][X]"));
					src_ids.insert(src_ids.end(),src_wids.begin()+nt1_beg+nt1_span+1,src_wids.begin()+nt2_beg);
					src_ids.push_back(src_vocab->get_id("[X][X]"));
					src_ids.insert(src_ids.end(),src_wids.begin()+nt2_beg+nt2_span+1,src_wids.begin()+beg+span+1);
					Pattern pattern;
					pattern.src_ids = src_ids;
					pattern.span_src_x1 = make_pair(nt1_beg,nt1_span);
					pattern.span_src_x2 = make_pair(nt2_beg,nt2_span);
					possible_patterns.push_back(pattern);
				}
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 获取当前跨度能匹配的glue pattern
 2. 入口参数: 当前跨度的起始位置和长度
 3. 出口参数: 能匹配的pattern
 4. 算法简介: 按照第一个非终结符的长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::get_patterns_for_glue_rule(const size_t beg,const size_t span,vector<Pattern> &possible_patterns)
{
	if (beg != 0 || span == 0)                                          //当前span不从句首开始或者只包含一个单词
		return;
	for (int nt1_span=0;nt1_span<span;nt1_span++)
	{
		vector<int> src_ids;
		src_ids.push_back(src_vocab->get_id("[X][X]"));
		src_ids.push_back(src_vocab->get_id("[X][X]"));
		Pattern pattern;
		pattern.src_ids = src_ids;
		pattern.span_src_x1 = make_pair(0,nt1_span);
		pattern.span_src_x2 = make_pair(nt1_span+1,span-nt1_span-1);
		possible_patterns.push_back(pattern);
	}
}

/**************************************************************************************
 1. 函数功能: 合并两个子候选并将生成的候选加入candpq_merge中
 2. 入口参数: 两个子候选,两个子候选的排名
 3. 出口参数: 更新后的candpq_merge
 4. 算法简介: 顺序以及逆序合并两个子候选
************************************************************************************* */
void SentenceTranslator::generate_cand_with_rule_and_add_to_pq(Rule &rule,int rank_x1,int rank_x2,Candpq &candpq_merge)
{
	if (rule.span_x2.first != -1)                                                                     //该规则有两个非终结符
	{
		Cand *cand_x1 = candbeam_matrix.at(rule.span_x1.first).at(rule.span_x1.second).at(rank_x1);
		Cand *cand_x2 = candbeam_matrix.at(rule.span_x2.first).at(rule.span_x2.second).at(rank_x2);
		Cand* cand = new Cand;
		cand->applied_rule = rule;
		cand->tgt_word_num = cand_x1->tgt_word_num + cand_x2->tgt_word_num + rule.tgt_rule.size() - 2;
		cand->rule_num = cand_x1->rule_num + cand_x2->rule_num + 1;
		cand->rank_x1 = rank_x1;
		cand->rank_x2 = rank_x2;
		cand->child_x1 = cand_x1;
		cand->child_x2 = cand_x2;
		int nonterminal_rank = 1;
		for (auto tgt_wid : rule.tgt_rule)
		{
			if (tgt_wid == tgt_vocab->get_id("[X][X]"))
			{
				if (nonterminal_rank == 1)
				{
					cand->tgt_wids.insert(cand->tgt_wids.end(),cand_x1->tgt_wids.begin(),cand_x1->tgt_wids.end());
					nonterminal_rank += 1;
				}
				else
				{
					cand->tgt_wids.insert(cand->tgt_wids.end(),cand_x1->tgt_wids.begin(),cand_x1->tgt_wids.end());
				}
			}
			else
			{
				cand->tgt_wids.push_back(tgt_wid);
			}
		}
		for (size_t i=0;i<PROB_NUM;i++)
		{
			cand->trans_probs.push_back(cand_x1->trans_probs.at(i)+cand_x2->trans_probs.at(i));    //TODO 还有规则的翻译概率
		}
		double increased_lm_prob = lm_model->cal_increased_lm_score(cand);
		cand->lm_prob = cand_x1->lm_prob + cand_x2->lm_prob + increased_lm_prob;
		cand->score = cand_x1->score + cand_x2->score 
			+ feature_weight.lm*increased_lm_prob;        //TODO 考虑每个特征的打分
		candpq_merge.push(cand);
	}
	else
	{
		Cand *cand_x1 = candbeam_matrix.at(rule.span_x1.first).at(rule.span_x1.second).at(rank_x1);
		Cand* cand = new Cand;
		cand->applied_rule = rule;
		cand->tgt_word_num = cand_x1->tgt_word_num + rule.tgt_rule.size() - 1;
		cand->rule_num = cand_x1->rule_num + 1;
		cand->rank_x1 = rank_x1;
		cand->rank_x2 = -1;
		cand->child_x1 = cand_x1;
		cand->child_x2 = NULL;
		for (auto tgt_wid : rule.tgt_rule)
		{
			if (tgt_wid == tgt_vocab->get_id("[X][X]"))
			{
				cand->tgt_wids.insert(cand->tgt_wids.end(),cand_x1->tgt_wids.begin(),cand_x1->tgt_wids.end());
			}
			else
			{
				cand->tgt_wids.push_back(tgt_wid);
			}
		}
		for (size_t i=0;i<PROB_NUM;i++)
		{
			cand->trans_probs.push_back(cand_x1->trans_probs.at(i));    //还有规则的翻译概率
		}
		double increased_lm_prob = lm_model->cal_increased_lm_score(cand);
		cand->lm_prob = cand_x1->lm_prob + increased_lm_prob;
		cand->score = cand_x1->score + feature_weight.lm*increased_lm_prob;        //TODO 考虑每个特征的打分
		candpq_merge.push(cand);
	}
}

/**************************************************************************************
 1. 函数功能: 将当前候选的邻居加入candpq_merge中
 2. 入口参数: 当前候选
 3. 出口参数: 更新后的candpq_merge
 4. 算法简介: a) 取比当前候选左子候选差一名的候选与当前候选的右子候选合并
              b) 取比当前候选右子候选差一名的候选与当前候选的左子候选合并
************************************************************************************* */
void SentenceTranslator::add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq_merge)
{
	if (cur_cand->rank_x2 != -1)                                                //如果生成当前候选的规则包括两个非终结符
	{
		int rank_x1 = cur_cand->rank_x1 + 1;
		int rank_x2 = cur_cand->rank_x2;
		if (candbeam_matrix.at(cur_cand->applied_rule.span_x1.first).at(cur_cand->applied_rule.span_x1.second).size() >= rank_x1)
		{
			generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge);
		}

		rank_x1 = cur_cand->rank_x1;
		rank_x2 = cur_cand->rank_x2 + 1;
		if (candbeam_matrix.at(cur_cand->applied_rule.span_x2.first).at(cur_cand->applied_rule.span_x2.second).size() >= rank_x2)
		{
			generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge);
		}
	}
	else
	{
		int rank_x1 = cur_cand->rank_x1 + 1;
		int rank_x2 = cur_cand->rank_x2;
		if (candbeam_matrix.at(cur_cand->applied_rule.span_x1.first).at(cur_cand->applied_rule.span_x1.second).size() >= rank_x1)
		{
			generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge);
		}
	}
}
