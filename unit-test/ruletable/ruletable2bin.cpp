#include "myutils.h"
const int LEN = 4096;

void ruletable2bin(string rule_filename)
{
	unordered_map <string,int> ch_vocab;
	unordered_map <string,int> en_vocab;
	vector<string> ch_vocab_vec;
	vector<string> en_vocab_vec;
	int ch_word_id = 0;
	int en_word_id = 0;
	gzFile gzfp = gzopen(rule_filename.c_str(),"r");
	if (!gzfp)
	{
		cout<<"fail to open "<<rule_filename<<endl;
		return;
	}
	ofstream fout;
	fout.open("prob.bin",ios::binary);
	if (!fout.is_open())
	{
		cout<<"fail open model file to write!\n";
		return;
	}
	char buf[LEN];
	while( gzgets(gzfp,buf,LEN) != Z_NULL)
	{
		string line(buf);
		vector <string> elements;
		string sep = "|||";
		Split(elements,line,sep);
		for (auto &e : elements)
		{
			TrimLine(e);
		}
		vector <string> ch_word_vec;
		Split(ch_word_vec,elements[0]);
		ch_word_vec.pop_back();
		vector <int> ch_id_vec;
		for (const auto &ch_word : ch_word_vec)
		{
			auto it = ch_vocab.find(ch_word);
			if (it != ch_vocab.end())
			{
				ch_id_vec.push_back(it->second);
			}
			else
			{
				ch_id_vec.push_back(ch_word_id);
				ch_vocab.insert(make_pair(ch_word,ch_word_id));
				ch_vocab_vec.push_back(ch_word);
				ch_word_id++;
			}
		}

		vector<int> nonterminal_idx_en;
		int idx_en = -1;
		short int rule_type = 0;                     //规则类型，0和1表示规则包含0个或1个非终结符，2表示规则包含2个正序非终结符，3表示规则包含2个逆序非终结符
		vector <string> en_word_vec;
		Split(en_word_vec,elements[1]);
		en_word_vec.pop_back();
		vector <int> en_id_vec;
		for (const auto &en_word : en_word_vec)
		{
			idx_en += 1;
			if (en_word == "[X][X]")
			{
				nonterminal_idx_en.push_back(idx_en);
			}
			auto it = en_vocab.find(en_word);
			if (it != en_vocab.end())
			{
				en_id_vec.push_back(it->second);
			}
			else
			{
				en_id_vec.push_back(en_word_id);
				en_vocab.insert(make_pair(en_word,en_word_id));
				en_vocab_vec.push_back(en_word);
				en_word_id++;
			}
		}
		if (nonterminal_idx_en.size() == 1)
		{
			rule_type = 1;
		}

		vector <string> prob_str_vec;
		vector <double> prob_vec;
		Split(prob_str_vec,elements[2]);
		for (const auto &prob_str : prob_str_vec)
		{
			prob_vec.push_back(stod(prob_str));
		}

		if (nonterminal_idx_en.size() == 2)
		{
			vector <string> alignments;
			sep = "-";
			Split(alignments,elements[3]);
			for (auto &align_str : alignments)
			{
				vector <string> pos_pair;
				Split(pos_pair,align_str,sep);
				int idx_en = stoi(pos_pair[1]);
				if (idx_en == nonterminal_idx_en[0])
				{
					rule_type = 2;
					break;
				}
				else if (idx_en == nonterminal_idx_en[1])
				{
					rule_type = 3;
					break;
				}
			}
		}
		short int ch_rule_len = ch_id_vec.size();
		short int en_rule_len = en_id_vec.size();
		fout.write((char*)&ch_rule_len,sizeof(short int));
		fout.write((char*)&ch_id_vec[0],sizeof(int)*ch_rule_len);
		fout.write((char*)&en_rule_len,sizeof(short int));
		fout.write((char*)&en_id_vec[0],sizeof(int)*en_rule_len);
		fout.write((char*)&prob_vec[0],sizeof(double)*prob_vec.size());
		fout.write((char*)&rule_type,sizeof(short int));
		/*
		cout<<ch_rule_len<<' ';
		for (auto e: ch_id_vec)
			cout<<e<<' ';
		cout<<"||| "<<en_rule_len<<' ';
		for (auto e: en_id_vec)
			cout<<e<<' ';
		cout<<"||| ";
		for (auto e: prob_vec)
			cout<<e<<' ';
		cout<<"||| "<<rule_type<<endl;
		*/
	}
	short int ch_rule_len = 2; 											//写入glue规则
	vector<int> ch_id_vec = {ch_vocab["[X][X]"],ch_vocab["[X][X]"]};
	short int en_rule_len = 2;
	vector<int> en_id_vec = {en_vocab["[X][X]"],en_vocab["[X][X]"]};
	vector<double> prob_vec = {2.718,2.718,2.718,2.718};
	short int rule_type = 2;
	fout.write((char*)&ch_rule_len,sizeof(short int));
	fout.write((char*)&ch_id_vec[0],sizeof(int)*ch_rule_len);
	fout.write((char*)&en_rule_len,sizeof(short int));
	fout.write((char*)&en_id_vec[0],sizeof(int)*en_rule_len);
	fout.write((char*)&prob_vec[0],sizeof(double)*prob_vec.size());
	fout.write((char*)&rule_type,sizeof(short int));
	gzclose(gzfp);
	fout.close();

	ofstream f_ch_vocab("vocab.ch");
	if (!f_ch_vocab.is_open())
	{
		cout<<"fail open ch vocab file to write!\n";
		return;
	}
	for(size_t i=0;i<ch_vocab_vec.size();i++)
	{
		f_ch_vocab<<ch_vocab_vec.at(i)+" "+to_string(i)+"\n";
	}
	f_ch_vocab.close();

	ofstream f_en_vocab("vocab.en");
	if (!f_en_vocab.is_open())
	{
		cout<<"fail open en vocab file to write!\n";
		return;
	}
	for(size_t i=0;i<en_vocab_vec.size();i++)
	{
		f_en_vocab<<en_vocab_vec.at(i)+" "+to_string(i)+"\n";
	}
	f_en_vocab.close();
}

int main(int argc,char* argv[])
{
    if(argc == 1)
    {
		cout<<"usage: ./ruletable2bin ruletable.gz\n";
		return 0;
    }
    string rule_filename(argv[1]);
    ruletable2bin(rule_filename);
	return 0;
}
