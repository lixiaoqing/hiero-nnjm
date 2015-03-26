#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <map>
using namespace std;

vector<int> src_wids = {1,2,3,4,5};

void get_patterns_with_one_terminal(const size_t beg,const size_t span,vector<vector<int> > &possible_patterns)
{
	if (span == 0)                                          //当前span只包含一个单词
		return;
	for (int nt_beg=beg;nt_beg<beg+span+1;nt_beg++)
	{
		for (int nt_span=0;nt_span<beg+span+1-nt_beg && nt_span<span;nt_span++)
		{
			vector<int> pattern;
			pattern.insert(pattern.end(),src_wids.begin()+beg,src_wids.begin()+nt_beg);
			pattern.push_back(0);
			pattern.insert(pattern.end(),src_wids.begin()+nt_beg+nt_span+1,src_wids.begin()+beg+span+1);
			possible_patterns.push_back(pattern);
		}
	}
}

void get_patterns_with_two_terminals(const size_t beg,const size_t span,vector<vector<int> > &possible_patterns)
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
					vector<int> pattern;
					pattern.insert(pattern.end(),src_wids.begin()+beg,src_wids.begin()+beg+nt1_beg);
					pattern.push_back(0);
					pattern.insert(pattern.end(),src_wids.begin()+nt1_beg+nt1_span+1,src_wids.begin()+nt2_beg);
					pattern.push_back(0);
					pattern.insert(pattern.end(),src_wids.begin()+nt2_beg+nt2_span+1,src_wids.begin()+beg+span+1);
					possible_patterns.push_back(pattern);
				}
			}
		}
	}
}

int main()
{
	vector<vector<int> > patterns;
	get_patterns_with_two_terminals(0,4,patterns);
	for (auto &pattern : patterns)
	{
		for (auto i : pattern)
			cout<<i<<' ';
		cout<<endl;
	}
}


