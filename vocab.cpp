#include "vocab.h"

void Vocab::load_vocab(const string &vocab_file)
{
	ifstream fin(vocab_file.c_str());
	if (!fin.is_open())
	{
		cerr<<"cannot open vocab file!\n";
		return;
	}
	string line;
	while(getline(fin,line))
	{
		istringstream buff(line);
		string word;
		int index;
		if(!(buff>>word>>index))
		{
			cerr<<"reading word and index error!"<<endl;
			return;
		}
		word_list.push_back(word);
		word2id.insert(make_pair(word,index));
	}
}

int Vocab::get_id(const string &word)
{
	auto it=word2id.find(word);
	if (it != word2id.end())
	{
		return it->second;
	}
	else
	{
		int id = word_list.size();
		word2id.insert(make_pair(word,id));
		word_list.push_back(word);
		return id;
	}
}

