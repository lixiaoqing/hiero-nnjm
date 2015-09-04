#include "myutils.h"
const int LEN = 4096;

bool load_block(vector<string> &data_block, gzFile &gzfp,int block_size)
{
    data_block.clear();
	int num = 0;
	string line;
	char buf[LEN];
	while( gzgets(gzfp,buf,LEN) != Z_NULL)
	{
		string line(buf);
		data_block.push_back(line);
		num++;
		if (num == block_size)
            break;
	}
    if (data_block.empty())
        return false;
    return true;
}

void extract_vocab(string rule_filename, unordered_map<string,int> &ch_vocab, unordered_map<string,int> &en_vocab)
{
	vector<string> ch_vocab_vec;
	vector<string> en_vocab_vec;
	int ch_word_id = 0;
	int en_word_id = 0;
	gzFile gzfp = gzopen(rule_filename.c_str(),"r");
	if (!gzfp)
	{
		cout<<"fail to open "<<rule_filename<<endl;
		exit(0);
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
		for (const auto &ch_word : ch_word_vec)
		{
			if (ch_vocab.find(ch_word) == ch_vocab.end())
			{
				ch_vocab.insert(make_pair(ch_word,ch_word_id));
				ch_vocab_vec.push_back(ch_word);
				ch_word_id++;
			}
		}

		vector <string> en_word_vec;
		Split(en_word_vec,elements[1]);
		en_word_vec.pop_back();
		for (const auto &en_word : en_word_vec)
		{
			if (en_vocab.find(en_word) == en_vocab.end())
			{
				en_vocab.insert(make_pair(en_word,en_word_id));
				en_vocab_vec.push_back(en_word);
				en_word_id++;
			}
		}
    }
	gzclose(gzfp);

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

void ruletable2bin(string rule_filename)
{
	unordered_map <string,int> ch_vocab;
	unordered_map <string,int> en_vocab;
    extract_vocab(rule_filename,ch_vocab,en_vocab);

	ofstream fout;
	fout.open("prob.bin",ios::binary);
	if (!fout.is_open())
	{
		cout<<"fail open model file to write!\n";
		exit(0);
	}

	gzFile gzfp = gzopen(rule_filename.c_str(),"r");
    vector<string> data_block;
    int block_size = 1000;
	while(load_block(data_block,gzfp,block_size))
    {
        vector<vector<int> > ch_id_vec_list(block_size,vector<int>());
        vector<vector<int> > en_id_vec_list(block_size,vector<int>());
        vector<vector<int> > en_to_ch_idx_list(block_size,vector<int>());
        vector<vector<double> > prob_vec_list(block_size,vector<double>());
        vector<short int> rule_type_list(block_size,0);

#pragma omp parallel for num_threads(16)
        for (int i=0;i<data_block.size();i++)
        {
            string line = data_block.at(i);
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
                ch_id_vec.push_back(ch_vocab[ch_word]);
            }

            vector<int> nonterminal_idx_en;
            int idx_en = -1;
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
                en_id_vec.push_back(en_vocab[en_word]);
            }

            vector <string> prob_str_vec;
            vector <double> prob_vec;
            Split(prob_str_vec,elements[2]);
            for (const auto &prob_str : prob_str_vec)
            {
                double prob = stod(prob_str);
                double log_prob = 0.0;
                if( abs(prob) <= numeric_limits<double>::epsilon() )
                {
                    log_prob = LogP_PseudoZero;
                }
                else
                {
                    log_prob = log10(prob);
                }
                prob_vec.push_back(log_prob);
            }

            short int rule_type;                        //规则类型，0和1表示包含0或1个非终结符，2和3表示正序和逆序hiero规则，4表示glue规则
            if (nonterminal_idx_en.empty())
            {
                rule_type = 0;
            }
            else if (nonterminal_idx_en.size() == 1)
            {
                rule_type = 1;
            }
            else
            {
                rule_type = 2;
            }

            vector <string> alignments;
            Split(alignments,elements[3]);
            vector<vector<int> > en_to_ch_idx_vec(en_id_vec.size(),vector<int>());
            vector<int> en_to_ch_idx(en_id_vec.size(),-99);
            sep = "-";
            nonterminal_idx_en.resize(2,-1);            //防止越界
            bool flag = true;
            for (auto &align_str : alignments)
            {
                vector <string> idx_pair;
                Split(idx_pair,align_str,sep);
                int idx_ch = stoi(idx_pair[0]);
                int idx_en = stoi(idx_pair[1]);

                if (idx_en == nonterminal_idx_en[0])
                {
                    en_to_ch_idx_vec.at(idx_en).push_back(-1);
                    flag = false;
                }
                else if (idx_en == nonterminal_idx_en[1])
                {
                    en_to_ch_idx_vec.at(idx_en).push_back(-2);
                    if (flag == true)
                    {
                        rule_type = 3;
                        flag = false;
                    }
                }
                else
                {
                    en_to_ch_idx_vec.at(idx_en).push_back(idx_ch);
                }
            }
            for (int j=0;j<en_to_ch_idx_vec.size();j++)
            {
                auto &ch_idx_vec = en_to_ch_idx_vec.at(j);
                if (ch_idx_vec.empty())
                    continue;
                if (ch_idx_vec.front() < 0)
                {
                    en_to_ch_idx.at(j) = ch_idx_vec.front();
                }
                else
                {
                    en_to_ch_idx.at(j) = (*min_element(ch_idx_vec.begin(),ch_idx_vec.end()) + *max_element(ch_idx_vec.begin(),ch_idx_vec.end()))/2;
                }
            }

            ch_id_vec_list.at(i) = ch_id_vec;
            en_id_vec_list.at(i) = en_id_vec;
            en_id_vec_list.at(i) = en_id_vec;
            en_to_ch_idx_list.at(i) = en_to_ch_idx;
            prob_vec_list.at(i) = prob_vec;
            rule_type_list.at(i) = rule_type;
        }
        for (int i=0; i<ch_id_vec_list.size();i++)
        {
            if (ch_id_vec_list[i].empty())
                break;
            short int ch_rule_len = ch_id_vec_list[i].size();
            short int en_rule_len = en_id_vec_list[i].size();
            fout.write((char*)&ch_rule_len,sizeof(short int));
            fout.write((char*)&ch_id_vec_list[i][0],sizeof(int)*ch_rule_len);
            fout.write((char*)&en_rule_len,sizeof(short int));
            fout.write((char*)&en_id_vec_list[i][0],sizeof(int)*en_rule_len);
            fout.write((char*)&en_to_ch_idx_list[i][0],sizeof(int)*en_rule_len);
            fout.write((char*)&prob_vec_list[i][0],sizeof(double)*prob_vec_list[i].size());
            fout.write((char*)&rule_type_list[i],sizeof(short int));
        }
    }
	gzclose(gzfp);

	short int ch_rule_len = 2; 											//写入glue规则
	vector<int> ch_id_vec = {ch_vocab["[X][X]"],ch_vocab["[X][X]"]};
	short int en_rule_len = 2;
	vector<int> en_id_vec = {en_vocab["[X][X]"],en_vocab["[X][X]"]};
    vector<int> en_to_ch_idx = {0,1};
	vector<double> prob_vec = {0,0,0,0};
	short int rule_type = 4;
	fout.write((char*)&ch_rule_len,sizeof(short int));
	fout.write((char*)&ch_id_vec[0],sizeof(int)*ch_rule_len);
	fout.write((char*)&en_rule_len,sizeof(short int));
	fout.write((char*)&en_id_vec[0],sizeof(int)*en_rule_len);
    fout.write((char*)&en_to_ch_idx[0],sizeof(int)*en_rule_len);
	fout.write((char*)&prob_vec[0],sizeof(double)*prob_vec.size());
	fout.write((char*)&rule_type,sizeof(short int));
	fout.close();
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

