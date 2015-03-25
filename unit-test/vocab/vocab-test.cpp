#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <map>
using namespace std;

int main()
{
	vector<string> word_list;
	map<string,int> word2id;
	ifstream fin("vocab.en");
	if (!fin.is_open())
	{
		cerr<<"cannot open dict file!\n";
		return 0;
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
			return 0;
		}
		cout<<word<<' '<<index+1<<endl;
		word_list.push_back(word);
		word2id.insert(make_pair(word,index));
	}
	cout<<"load over\n";
}

