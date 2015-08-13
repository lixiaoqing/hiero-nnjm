#include "cand.h"

bool larger( const Cand *pl, const Cand *pr )
{
	return pl->score > pr->score;
}

bool smaller( const Cand *pl, const Cand *pr )
{
	return pl->score < pr->score;
}

/************************************************************************
 1. 函数功能: 将翻译候选加入列表中, 并进行假设重组
 2. 入口参数: 翻译候选的指针
 3. 出口参数: 如果候选被丢弃或者替换掉原来的候选,返回false;否则返回true
 4. 算法简介: a) 如果当前候选与优先级队列中的某个候选的目标端边界词相同,
              a.1) 如果当前候选的得分低, 则丢弃当前候选
              a.2) 如果当前候选的得分低, 则替换原候选
              a.3) 如果二者得分相同, 则将当前候选加入列表
              b) 如果当前候选与优先级队列中的所有候选的目标端边界词不同,
	         则将当前候选加入列表
 * **********************************************************************/
void CandBeam::add(Cand *&cand_ptr,int beam_size)
{ 
	for (auto &e_cand_ptr : data)
	{
		if (is_bound_same(cand_ptr,e_cand_ptr))
		{
			if (cand_ptr->score > e_cand_ptr->score)
			{
				swap(e_cand_ptr,cand_ptr);
			}
			delete cand_ptr;
			return;
		}
	}
	if (data.size() >= beam_size)
	{
        auto &min_cand_ptr = *min_element(data.begin(),data.end(),smaller);
        if (cand_ptr->score > min_cand_ptr->score)
        {
            swap(min_cand_ptr,cand_ptr);
        }
		delete cand_ptr;
		return;
	}
	data.push_back(cand_ptr); 
}

bool CandBeam::is_bound_same(const Cand *a, const Cand *b)
{
	size_t len_a = a->tgt_wids.size();
	size_t len_b = b->tgt_wids.size();
	size_t bound_len_a = min(len_a, LM_ORDER-1);
	size_t bound_len_b = min(len_b, LM_ORDER-1);
	if (bound_len_a != bound_len_b)
		return false;
	if (bound_len_a < LM_ORDER && a->tgt_wids != b->tgt_wids)
		return false;
	for (size_t i=0;i<bound_len_a;i++)
	{
		if (a->tgt_wids.at(i) != b->tgt_wids.at(i) || a->tgt_wids.at(len_a-1-i) != b->tgt_wids.at(len_b-1-i))
			return false;
	}
	return true;
}

void CandBeam::free()
{
	for (auto cand : data)
	{
		delete cand;
	}
}
