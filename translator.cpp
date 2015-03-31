#include "translator.h"

SentenceTranslator::SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen)
{
	src_vocab = i_models.src_vocab;
	tgt_vocab = i_models.tgt_vocab;
	ruletable = i_models.ruletable;
	lm_model = i_models.lm_model;
	para = i_para;
	feature_weight = i_weight;

	src_nt_id = src_vocab->get_id("[X][X]");
	tgt_nt_id = tgt_vocab->get_id("[X][X]");
	stringstream ss(input_sen);
	string word;
	while(ss>>word)
	{
		src_wids.push_back(src_vocab->get_id(word));
	}

	src_sen_len = src_wids.size();
	span2cands.resize(src_sen_len);
	span2rules.resize(src_sen_len);
	for (size_t beg=0;beg<src_sen_len;beg++)
	{
		span2cands.at(beg).resize(src_sen_len-beg);
		span2rules.at(beg).resize(src_sen_len-beg);
	}

	fill_span2cands_with_phrase_rules();
	get_applicable_rules_for_each_span();
}

SentenceTranslator::~SentenceTranslator()
{
	for (size_t i=0;i<span2cands.size();i++)
	{
		for(size_t j=0;j<span2cands.at(i).size();j++)
		{
			span2cands.at(i).at(j).free();
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
		//cout<<"find matched phrase rules over\n";
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
					cand->applied_rule.src_ids.push_back(src_wids.at(beg));
					cand->lm_prob = lm_model->cal_increased_lm_score(cand);
					cand->score += feature_weight.rule_num*cand->rule_num 
						       + feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
					span2cands.at(beg).at(span).add(cand);
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
				cand->applied_rule.tgt_rule = &tgt_rule;
				cand->lm_prob = lm_model->cal_increased_lm_score(cand);
				cand->score += feature_weight.rule_num*cand->rule_num 
					       + feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
				span2cands.at(beg).at(span).add(cand);
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 找到每个跨度所有能用的hiero规则，并加入到rules_matrix中
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: 1) 找出当前句子所有可能的pattern，以及每个pattern对应的所有跨度
 			  2) 对每个pattern，检查规则表中是否存在可用的规则
 			  3) 根据每个可用的规则更新span2rules
************************************************************************************* */
void SentenceTranslator::get_applicable_rules_for_each_span()
{
	vector<Pattern> possible_patterns;
	get_patterns_with_one_terminal_seq(possible_patterns);            //形如AX,XA和XAX的pattern
	get_patterns_with_two_terminal_seq(possible_patterns);            //形如AXB,AXBX和XAXB的pattern
	get_patterns_with_three_terminal_seq(possible_patterns);          //形如AXBXC的pattern
	get_patterns_for_glue_rule(possible_patterns);                    //起始位置为句首，形如X1X2的pattern

	for (auto &pattern : possible_patterns)
	{
		vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(pattern.src_ids,0);
		if (matched_rules_for_prefixes.size() == pattern.src_ids.size() && matched_rules_for_prefixes.back() != NULL)         //找到了可用的规则
		{
			for (auto &tgt_rule : *matched_rules_for_prefixes.back())
			{
				for (auto &pattern_span : pattern.pattern_spans)
				{
					Rule rule;
					rule.src_ids = pattern.src_ids;
					rule.tgt_rule = &tgt_rule;
					if (tgt_rule.rule_type == 3)
					{
						rule.span_x1 = pattern_span.span_src_x2;
						rule.span_x2 = pattern_span.span_src_x1;
					}
					else
					{
						rule.span_x1 = pattern_span.span_src_x1;
						rule.span_x2 = pattern_span.span_src_x2;
					}
					span2rules.at(pattern_span.span.first).at(pattern_span.span.second).push_back(rule);
				}
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
	CandBeam &candbeam = span2cands.at(0).at(src_sen_len-1);
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
	Cand *best_cand = span2cands.at(0).at(src_sen_len-1).top();
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
	for (auto src_wid : cand->applied_rule.src_ids)
	{
		rule += src_vocab->get_word(src_wid)+" ";
	}
	rule += "||| ";
	for (auto tgt_wid : cand->applied_rule.tgt_rule->wids)
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
		span2cands.at(beg).at(0).sort();		               //对列表中的候选进行排序
	}
	for (size_t span=1;span<src_sen_len;span++)
	{
#pragma omp parallel for num_threads(para.SPAN_THREAD_NUM)
		for(size_t beg=0;beg<src_sen_len-span;beg++)
		{
			generate_kbest_for_span(beg,span);
			span2cands.at(beg).at(span).sort();
		}
	}
	return words_to_str(span2cands.at(0).at(src_sen_len-1).top()->tgt_wids,true);
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

	//对于当前跨度匹配到的每一条规则,取出非终结符对应的跨度中的最好候选,将合并得到的候选加入candpq_merge
	for(auto &rule : span2rules.at(beg).at(span))
	{
		generate_cand_with_rule_and_add_to_pq(rule,0,0,candpq_merge);
	}

	//立方体剪枝,每次从candpq_merge中取出最好的候选加入span2cands中,并将该候选的邻居加入candpq_merge中
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
		bool flag = span2cands.at(beg).at(span).add(best_cand);
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
 1. 函数功能: 获取当前句子能匹配的所有包含一个终结符序列的pattern
 2. 入口参数: 无
 3. 出口参数: 能匹配的pattern
 4. 算法简介: 按照终结符序列的起始位置和长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::get_patterns_with_one_terminal_seq(vector<Pattern> &possible_patterns)
{
	for (int ts_beg=0;ts_beg<src_sen_len;ts_beg++)
	{
		for (int ts_span=0;ts_span<src_sen_len-ts_beg && ts_span<SPAN_LEN_MAX;ts_span++)
		{
			vector<int> src_ts_ids(src_wids.begin()+ts_beg,src_wids.begin()+ts_beg+ts_span+1);
			//抽取形如XA的pattern
			if (ts_beg != 0)
			{
				Pattern pattern;
				pattern.src_ids.push_back(src_nt_id);
				pattern.src_ids.insert(pattern.src_ids.end(),src_ts_ids.begin(),src_ts_ids.end());
				for (int nt_span=0;nt_span<ts_beg && nt_span<SPAN_LEN_MAX-ts_span-1;nt_span++)   //TODO 注意边界取值
				{
					PatternSpan pattern_span;
					pattern_span.span = make_pair(ts_beg-nt_span-1,ts_span+nt_span+1);
					pattern_span.span_src_x1 = make_pair(ts_beg-nt_span-1,nt_span);
					pattern_span.span_src_x2 = make_pair(-1,-1);
					pattern.pattern_spans.push_back(pattern_span);
				}
				possible_patterns.push_back(pattern);
			}
			//抽取形如AX的pattern
			if (ts_beg+ts_span != src_sen_len - 1)
			{
				Pattern pattern;
				pattern.src_ids = src_ts_ids;
				pattern.src_ids.push_back(src_nt_id);
				for (int nt_span=0;nt_span<src_sen_len-ts_beg-ts_span-1 && nt_span<SPAN_LEN_MAX-ts_span-1;nt_span++)   //TODO 注意边界取值
				{
					PatternSpan pattern_span;
					pattern_span.span = make_pair(ts_beg,ts_span+nt_span+1);
					pattern_span.span_src_x1 = make_pair(ts_beg+ts_span+1,nt_span);
					pattern_span.span_src_x2 = make_pair(-1,-1);
					pattern.pattern_spans.push_back(pattern_span);
				}
				possible_patterns.push_back(pattern);
			}
			//抽取形如XAX的pattern
			if (ts_beg != 0 && ts_beg+ts_span != src_sen_len - 1)
			{
				Pattern pattern;
				pattern.src_ids.push_back(src_nt_id);
				pattern.src_ids.insert(pattern.src_ids.end(),src_ts_ids.begin(),src_ts_ids.end());
				pattern.src_ids.push_back(src_nt_id);
				for (int nt1_span=0;nt1_span<ts_beg && nt1_span<SPAN_LEN_MAX-ts_span-2;nt1_span++)   //TODO 注意边界取值
				{
					for (int nt2_span=0;nt2_span<src_sen_len-ts_beg-ts_span-1 && nt2_span<SPAN_LEN_MAX-ts_span-nt1_span-1;nt2_span++)   //TODO 注意边界取值
					{
						PatternSpan pattern_span;
						pattern_span.span = make_pair(ts_beg-nt1_span-1,ts_span+nt1_span+nt2_span+1);
						pattern_span.span_src_x1 = make_pair(ts_beg-nt1_span-1,nt1_span);
						pattern_span.span_src_x2 = make_pair(ts_beg+ts_span+1,nt2_span);
						pattern.pattern_spans.push_back(pattern_span);
					}
				}
				possible_patterns.push_back(pattern);
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 获取当前句子能匹配的所有包含两个终结符序列的pattern
 2. 入口参数: 无
 3. 出口参数: 能匹配的pattern
 4. 算法简介: 按照终结符序列的起始位置和长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::get_patterns_with_two_terminal_seq(vector<Pattern> &possible_patterns)
{
	for (int ts_beg=0;ts_beg<src_sen_len;ts_beg++)
	{
		for (int ts_span=0;ts_span<src_sen_len-ts_beg && ts_span<SPAN_LEN_MAX;ts_span++)           //此处的ts_span为两个非终结符序列从头到尾的总跨度
		{
			for (int inner_nts_beg=ts_beg+1;inner_nts_beg<ts_beg+ts_span-1;inner_nts_beg++)
			{
				for (int inner_nts_span=0;inner_nts_span<ts_span-(inner_nts_beg-ts_beg);inner_nts_span++)
				{
					vector<int> src_ids(src_wids.begin()+ts_beg,src_wids.begin()+inner_nts_beg);
					src_ids.push_back(src_nt_id);
					src_ids.insert(src_ids.end(),src_wids.begin()+inner_nts_beg+inner_nts_span+1,src_wids.begin()+ts_beg+ts_span+1);
					//抽取形如XAXB的pattern
					if (ts_beg != 0)
					{
						Pattern pattern;
						pattern.src_ids.push_back(src_nt_id);
						pattern.src_ids.insert(pattern.src_ids.end(),src_ids.begin(),src_ids.end());
						for (int lhs_nt_span=0;lhs_nt_span<ts_beg && lhs_nt_span<SPAN_LEN_MAX-ts_span-1;lhs_nt_span++)   //TODO 注意边界取值
						{
							PatternSpan pattern_span;
							pattern_span.span = make_pair(ts_beg-lhs_nt_span-1,ts_span+lhs_nt_span+1);
							pattern_span.span_src_x1 = make_pair(ts_beg-lhs_nt_span-1,lhs_nt_span);
							pattern_span.span_src_x2 = make_pair(inner_nts_beg,inner_nts_span);
							pattern.pattern_spans.push_back(pattern_span);
						}
						possible_patterns.push_back(pattern);
					}
					//抽取形如AXBX的pattern
					if (ts_beg+ts_span != src_sen_len - 1)
					{
						Pattern pattern;
						pattern.src_ids = src_ids;
						pattern.src_ids.push_back(src_nt_id);
						for (int rhs_nt_span=0;rhs_nt_span<src_sen_len-ts_beg-ts_span-1 && rhs_nt_span<SPAN_LEN_MAX-ts_span-1;rhs_nt_span++)   //TODO 注意边界取值
						{
							PatternSpan pattern_span;
							pattern_span.span = make_pair(ts_beg,ts_span+rhs_nt_span+1);
							pattern_span.span_src_x1 = make_pair(inner_nts_beg,inner_nts_span);
							pattern_span.span_src_x2 = make_pair(ts_beg+ts_span+1,rhs_nt_span);
							pattern.pattern_spans.push_back(pattern_span);
						}
						possible_patterns.push_back(pattern);
					}
					//抽取形如AXB的pattern
					Pattern pattern;
					pattern.src_ids = src_ids;
					PatternSpan pattern_span;
					pattern_span.span = make_pair(ts_beg,ts_span);
					pattern_span.span_src_x1 = make_pair(inner_nts_beg,inner_nts_span);
					pattern_span.span_src_x2 = make_pair(-1,-1);
					pattern.pattern_spans.push_back(pattern_span);
					possible_patterns.push_back(pattern);
				}
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 获取当前句子能匹配的所有包含两个终结符序列的pattern
 2. 入口参数: 无
 3. 出口参数: 能匹配的pattern
 4. 算法简介: 按照终结符序列的起始位置和长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::get_patterns_with_three_terminal_seq(vector<Pattern> &possible_patterns)
{
	for (int ts_beg=0;ts_beg<src_sen_len;ts_beg++)
	{
		for (int ts_span=0;ts_span<src_sen_len-ts_beg && ts_span<SPAN_LEN_MAX;ts_span++)                    //此处的ts_span为三个非终结符序列从头到尾的总跨度
		{
			for (int inner_nts_beg=ts_beg+1;inner_nts_beg<ts_beg+ts_span-1;inner_nts_beg++)
			{
				for (int inner_nts_span=0;inner_nts_span<ts_span-(inner_nts_beg-ts_beg);inner_nts_span++)   //此处的inner_nts_span为两个非终结符序列从头到尾的总跨度
				{
					for (int inner_ts_beg=inner_nts_beg+1;inner_ts_beg<inner_nts_beg+inner_nts_span-1;inner_ts_beg++)
					{
						for (int inner_ts_span=0;inner_ts_span<inner_nts_span-(inner_ts_beg-inner_nts_beg);inner_ts_span++)
						{
							vector<int> src_ids(src_wids.begin()+ts_beg,src_wids.begin()+inner_nts_beg);
							src_ids.push_back(src_nt_id);
							src_ids.insert(src_ids.end(),src_wids.begin()+inner_ts_beg,src_wids.begin()+inner_ts_beg+inner_ts_span+1);
							src_ids.push_back(src_nt_id);
							src_ids.insert(src_ids.end(),src_wids.begin()+inner_ts_beg+inner_ts_span+1,src_wids.begin()+ts_beg+ts_span+1);
							//抽取形如AXBXC的pattern
							Pattern pattern;
							pattern.src_ids = src_ids;
							PatternSpan pattern_span;
							pattern_span.span = make_pair(ts_beg,ts_span);
							pattern_span.span_src_x1 = make_pair(inner_nts_beg,inner_ts_beg-inner_nts_beg-1);
							pattern_span.span_src_x2 = make_pair(inner_ts_beg+inner_ts_span+1,inner_nts_span-inner_ts_span-(inner_ts_beg-inner_nts_beg-1)-2);
							pattern.pattern_spans.push_back(pattern_span);
							possible_patterns.push_back(pattern);
						}
					}
				}
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 获取当前句子能匹配的glue pattern
 2. 入口参数: 无
 3. 出口参数: 能匹配的pattern
 4. 算法简介: 按照第一个非终结符的长度遍历所有可能的pattern
************************************************************************************* */
void SentenceTranslator::get_patterns_for_glue_rule(vector<Pattern> &possible_patterns)
{
	vector<int> src_ids = {src_nt_id,src_nt_id};
	Pattern pattern;
	pattern.src_ids = src_ids;
	for (int span=1;span<src_sen_len;span++)                      //glue pattern的跨度不受规则最大跨度RULE_LEN_MAX的限制，可以延伸到句尾
	{
		for (int nt1_span=0;nt1_span<span;nt1_span++)
		{
			PatternSpan pattern_span;
			pattern_span.span = make_pair(0,span);
			pattern_span.span_src_x1 = make_pair(0,nt1_span);
			pattern_span.span_src_x2 = make_pair(nt1_span+1,span-nt1_span-1);
			pattern.pattern_spans.push_back(pattern_span);
		}
	}
	possible_patterns.push_back(pattern);
}

/**************************************************************************************
 1. 函数功能: 合并两个子候选并将生成的候选加入candpq_merge中
 2. 入口参数: 两个子候选,两个子候选的排名
 3. 出口参数: 更新后的candpq_merge
 4. 算法简介: 顺序以及逆序合并两个子候选
************************************************************************************* */
void SentenceTranslator::generate_cand_with_rule_and_add_to_pq(Rule &rule,int rank_x1,int rank_x2,Candpq &candpq_merge)
{
	if (rule.tgt_rule->rule_type >= 2)                                                                      //该规则有两个非终结符
	{
		if (span2cands.at(rule.span_x1.first).at(rule.span_x1.second).size() <= rank_x1 ||
			span2cands.at(rule.span_x2.first).at(rule.span_x2.second).size() <= rank_x2)               //子候选不够用
			return;
		Cand *cand_x1 = span2cands.at(rule.span_x1.first).at(rule.span_x1.second).at(rank_x1);
		Cand *cand_x2 = span2cands.at(rule.span_x2.first).at(rule.span_x2.second).at(rank_x2);
		Cand* cand = new Cand;
		cand->applied_rule = rule;
		if (rule.tgt_rule->rule_type == 4)  //glue规则
		{
			cand->rule_num = cand_x1->rule_num + cand_x2->rule_num;
			cand->glue_num = cand_x1->glue_num + cand_x2->glue_num + 1;
		}
		else
		{
			cand->rule_num = cand_x1->rule_num + cand_x2->rule_num + 1;
			cand->glue_num = cand_x1->glue_num + cand_x2->glue_num;
		}
		cand->rank_x1 = rank_x1;
		cand->rank_x2 = rank_x2;
		cand->child_x1 = cand_x1;
		cand->child_x2 = cand_x2;
		cand->tgt_word_num = cand_x1->tgt_word_num + cand_x2->tgt_word_num + rule.tgt_rule->wids.size() - 2;
		int nt_idx = 1; 							//表示第几个非终结符
		for (auto tgt_wid : rule.tgt_rule->wids)
		{
			if (tgt_wid == tgt_nt_id)
			{
				if (nt_idx == 1)
				{
					cand->tgt_wids.insert(cand->tgt_wids.end(),cand_x1->tgt_wids.begin(),cand_x1->tgt_wids.end());
					nt_idx += 1;
				}
				else
				{
					cand->tgt_wids.insert(cand->tgt_wids.end(),cand_x2->tgt_wids.begin(),cand_x2->tgt_wids.end());
				}
			}
			else
			{
				cand->tgt_wids.push_back(tgt_wid);
			}
		}
		for (size_t i=0;i<PROB_NUM;i++)
		{
			cand->trans_probs.push_back(cand_x1->trans_probs.at(i) + cand_x2->trans_probs.at(i) + rule.tgt_rule->probs.at(i));
		}
		double increased_lm_prob = lm_model->cal_increased_lm_score(cand);
		cand->lm_prob = cand_x1->lm_prob + cand_x2->lm_prob + increased_lm_prob;
		cand->score = cand_x1->score + cand_x2->score + rule.tgt_rule->score + feature_weight.lm*increased_lm_prob
					  + feature_weight.rule_num*1 + feature_weight.len*(rule.tgt_rule->wids.size() - 2);
		if (rule.tgt_rule->rule_type == 4)  //glue规则
		{
			cand->score += feature_weight.glue*1;
		}
		candpq_merge.push(cand);
	}
	else 																							   //该规则只有一个非终结符
	{
		if (span2cands.at(rule.span_x1.first).at(rule.span_x1.second).size() <= rank_x1)
			return;
		Cand *cand_x1 = span2cands.at(rule.span_x1.first).at(rule.span_x1.second).at(rank_x1);
		Cand* cand = new Cand;
		cand->applied_rule = rule;
		cand->rule_num = cand_x1->rule_num + 1;
		cand->glue_num = cand_x1->glue_num;
		cand->rank_x1 = rank_x1;
		cand->rank_x2 = -1;
		cand->child_x1 = cand_x1;
		cand->child_x2 = NULL;
		cand->tgt_word_num = cand_x1->tgt_word_num + rule.tgt_rule->wids.size() - 1;
		for (auto tgt_wid : rule.tgt_rule->wids)
		{
			if (tgt_wid == tgt_nt_id)
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
			cand->trans_probs.push_back(cand_x1->trans_probs.at(i) + rule.tgt_rule->probs.at(i));
		}
		double increased_lm_prob = lm_model->cal_increased_lm_score(cand);
		cand->lm_prob = cand_x1->lm_prob + increased_lm_prob;
		cand->score = cand_x1->score + rule.tgt_rule->score + feature_weight.lm*increased_lm_prob
					  + feature_weight.rule_num*1 + feature_weight.len*(rule.tgt_rule->wids.size() - 1);
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
		generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge);

		rank_x1 = cur_cand->rank_x1;
		rank_x2 = cur_cand->rank_x2 + 1;
		generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge);
	}
	else 																		//如果生成当前候选的规则包括一个非终结符
	{
		int rank_x1 = cur_cand->rank_x1 + 1;
		int rank_x2 = cur_cand->rank_x2;
		generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,rank_x1,rank_x2,candpq_merge);
	}
}
