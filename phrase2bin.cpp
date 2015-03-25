#include "myutils.h"
const int LEN = 4096;

void phrase2bin(string phrase_filename,string mode)
{
	unordered_map <string,int> ch_vocab;
	unordered_map <string,int> en_vocab;
	vector<string> ch_vocab_vec;
	vector<string> en_vocab_vec;
	int ch_word_id = 0;
	int en_word_id = 0;
	gzFile gzfp = gzopen(phrase_filename.c_str(),"r");
	if (!gzfp)
	{
		cout<<"fail to open "<<phrase_filename<<endl;
		return;
	}
	ofstream fout;
	if (mode == "1")
		fout.open("prob.bin-with-alignment",ios::binary);
	else
		fout.open("prob.bin-without-alignment",ios::binary);
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

		vector <string> en_word_vec;
		Split(en_word_vec,elements[1]);
		vector <int> en_id_vec;
		for (const auto &en_word : en_word_vec)
		{
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

		vector <string> prob_str_vec;
		vector <double> prob_vec;
		Split(prob_str_vec,elements[2]);
		for (const auto &prob_str : prob_str_vec)
		{
			prob_vec.push_back(stod(prob_str));
		}

		vector <string> alignments;
		vector <int> alignment_vec;
		sep = "-";
		if (mode == "1")
		{
			Split(alignments,elements[3]);
			for (auto &align_str : alignments)
			{
				vector <string> pos_pair;
				Split(pos_pair,align_str,sep);
				alignment_vec.push_back(stoi(pos_pair[0]));
				alignment_vec.push_back(stoi(pos_pair[1]));
			}
		}
		short int ch_phrase_len = ch_id_vec.size();
		short int en_phrase_len = en_id_vec.size();
		short int alignment_vec_len = alignment_vec.size();
		fout.write((char*)&ch_phrase_len,sizeof(short int));
		fout.write((char*)&ch_id_vec[0],sizeof(int)*ch_phrase_len);
		fout.write((char*)&en_phrase_len,sizeof(short int));
		fout.write((char*)&en_id_vec[0],sizeof(int)*en_phrase_len);
		fout.write((char*)&prob_vec[0],sizeof(double)*prob_vec.size());
		if (mode == "1")
		{
			fout.write((char*)&alignment_vec_len,sizeof(short int));
			fout.write((char*)&alignment_vec[0],sizeof(int)*alignment_vec.size());
		}
	}
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
		cout<<"usage: ./phrase2bin phrase.gz mode\nconvert alignment if mode==1, don't convert if mode==0\n";
		return 0;
    }
    string phrase_filename(argv[1]);
	string mode(argv[2]);
    phrase2bin(phrase_filename,mode);
	return 0;
}
