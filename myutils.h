#include "stdafx.h"

void TrimLine(string &line);
void Split(vector<string> &vs, string &s);
void Split(vector<string> &vs, string &s, string &sep);
void load_data_into_blocks(vector<vector<string> > &data_blocks, ifstream &fin,int block_size);
