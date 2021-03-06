#include "translator.h"

void read_config(Filenames &fns,Parameter &para, Weight &weight, const string &config_file)
{
	ifstream fin;
	fin.open(config_file.c_str());
	if (!fin.is_open())
	{
		cerr<<"fail to open config file\n";
		return;
	}
	string line;
	while(getline(fin,line))
	{
		TrimLine(line);
		if (line == "[input-file]")
		{
			getline(fin,line);
			fns.input_file = line;
		}
		else if (line == "[output-file]")
		{
			getline(fin,line);
			fns.output_file = line;
		}
		else if (line == "[nbest-file]")
		{
			getline(fin,line);
			fns.nbest_file = line;
		}
		else if (line == "[src-vocab-file]")
		{
			getline(fin,line);
			fns.src_vocab_file = line;
		}
		else if (line == "[tgt-vocab-file]")
		{
			getline(fin,line);
			fns.tgt_vocab_file = line;
		}
		else if (line == "[rule-table-file]")
		{
			getline(fin,line);
			fns.rule_table_file = line;
		}
		else if (line == "[lm-file]")
		{
			getline(fin,line);
			fns.lm_file = line;
		}
		else if (line == "[nnjm-file]")
		{
			getline(fin,line);
			fns.nnjm_file = line;
		}
		else if (line == "[BEAM-SIZE]")
		{
			getline(fin,line);
			para.BEAM_SIZE = stoi(line);
		}
		else if (line == "[CUBE-SIZE]")
		{
			getline(fin,line);
			para.CUBE_SIZE = stoi(line);
		}
		else if (line == "[SEN-THREAD-NUM]")
		{
			getline(fin,line);
			para.SEN_THREAD_NUM = stoi(line);
		}
		else if (line == "[SPAN-THREAD-NUM]")
		{
			getline(fin,line);
			para.SPAN_THREAD_NUM = stoi(line);
		}
		else if (line == "[NBEST-NUM]")
		{
			getline(fin,line);
			para.NBEST_NUM = stoi(line);
		}
		else if (line == "[RULE-NUM-LIMIT]")
		{
			getline(fin,line);
			para.RULE_NUM_LIMIT = stoi(line);
		}
		else if (line == "[PRINT-NBEST]")
		{
			getline(fin,line);
			para.PRINT_NBEST = stoi(line);
		}
		else if (line == "[DUMP-RULE]")
		{
			getline(fin,line);
			para.DUMP_RULE = stoi(line);
		}
		else if (line == "[DROP-OOV]")
		{
			getline(fin,line);
			para.DROP_OOV = stoi(line);
		}
		else if (line == "[weight]")
		{
			while(getline(fin,line))
			{
				if (line == "")
					break;
				stringstream ss(line);
				string feature;
				ss >> feature;
				if (feature.find("trans") != string::npos)
				{
					double w;
					ss>>w;
					weight.trans.push_back(w);
				}
				else if(feature == "len")
				{
					ss>>weight.len;
				}
				else if(feature == "lm")
				{
					ss>>weight.lm;
				}
				else if(feature == "rule-num")
				{
					ss>>weight.rule_num;
				}
				else if(feature == "glue")
				{
					ss>>weight.glue;
				}
				else if(feature == "nnjm")
				{
					ss>>weight.nnjm;
				}
			}
		}
	}
}

void parse_args(int argc, char *argv[],Filenames &fns,Parameter &para, Weight &weight)
{
	read_config(fns,para,weight,"config.ini");
	for( int i=1; i<argc; i++ )
	{
		string arg( argv[i] );
		if( arg == "-i" )
		{
			fns.input_file = argv[++i];
		}
	}
}

void translate_file(const Models &models, const Parameter &para, const Weight &weight, const Filenames &fns)
{
	ifstream fin(fns.input_file.c_str());
    ofstream fout(fns.output_file.c_str());
    ofstream fnbest(fns.nbest_file.c_str());
    ofstream frules("applied-rules.txt");
	if (!fin.is_open() || !fout.is_open() || !fnbest.is_open() || !frules.is_open() )
	{
		cerr<<"file open error!\n";
        exit(0);
	}

	vector<vector<string> > input_sen_blocks;
    int block_size = para.SEN_THREAD_NUM;
    load_data_into_blocks(input_sen_blocks,fin,block_size);

    vector<neuralLM*> nnjm_models;
    nnjm_models.resize(block_size,NULL);
    for (int i=0;i<block_size;i++)
    {
        nnjm_models[i] = new neuralLM();
        nnjm_models[i]->read(fns.nnjm_file);
    }

    int sen_id = -1;
	int block_num = input_sen_blocks.size();
	for (size_t i=0;i<block_num;i++)
    {
        block_size = input_sen_blocks.at(i).size();
        vector<vector<string> > output_paras;
        vector<vector<vector<TuneInfo> > > nbest_tune_info_lists;
        vector<vector<vector<string> > > applied_rules_lists;
        output_paras.resize(block_size);
        nbest_tune_info_lists.resize(block_size);
        applied_rules_lists.resize(block_size);
        for (auto line : input_sen_blocks.at(i))
        {
            vector<string> vs;
            Split(vs,line);
            for (const auto &word : vs)
            {
                models.src_vocab->get_id(word);                                             //避免并行时同时修改vocab发生冲突
            }
        }
#pragma omp parallel for num_threads(block_size)
        for (size_t j=0;j<block_size;j++)
        {
            Models cur_models = models;
            cur_models.nnjm_model = nnjm_models.at(j);
            SentenceTranslator sen_translator(cur_models,para,weight,input_sen_blocks.at(i).at(j));
            output_paras.at(j) = sen_translator.translate_sentence();
            if (para.PRINT_NBEST == true)
            {
                nbest_tune_info_lists.at(j) = sen_translator.get_tune_info();
            }
            if (para.DUMP_RULE == true)
            {
                applied_rules_lists.at(j) = sen_translator.get_applied_rules();
            }
        }
        for (const auto &output_sens : output_paras)
        {
            for (const auto & sen : output_sens)
            {
                fout<<sen<<endl;
                //cout<<sen<<endl;
            }
        }
        if (para.PRINT_NBEST == true)
        {
            for (const auto &nbest_tune_info_list : nbest_tune_info_lists)
            {
                for (const auto &nbest_tune_info : nbest_tune_info_list)
                {
                    sen_id++;
                    for (const auto &tune_info : nbest_tune_info)
                    {
                        fnbest<<sen_id<<" ||| "<<tune_info.translation<<" ||| ";
                        for (const auto &v : tune_info.feature_values)
                        {
                            fnbest<<v<<' ';
                        }
                        fnbest<<"||| "<<tune_info.total_score<<endl;
                    }
                }
            }
        }
        if (para.DUMP_RULE == true)
        {
            for (const auto &applied_rules_list : applied_rules_lists)
            {
                for (const auto &applied_rules : applied_rules_list)
                {
                    for (const auto &applied_rule : applied_rules)
                    {
                        frules<<applied_rule;
                    }
                    frules<<endl;
                }
            }
        }
    }
}

int main( int argc, char *argv[])
{
	clock_t a,b;
	a = clock();

	omp_set_nested(1);
	Filenames fns;
	Parameter para;
	Weight weight;
	parse_args(argc,argv,fns,para,weight);

	Vocab *src_vocab = new Vocab(fns.src_vocab_file);
	Vocab *tgt_vocab = new Vocab(fns.tgt_vocab_file);
	RuleTable *ruletable = new RuleTable(para.RULE_NUM_LIMIT,weight,fns.rule_table_file,src_vocab,tgt_vocab);
	LanguageModel *lm_model = new LanguageModel(fns.lm_file,tgt_vocab);
    set<string> function_words;
	ifstream fin("data/function-words");
	if (!fin.is_open())
	{
		cerr<<"cannot open function words file!\n";
		exit(0);
	}
    string w;
	while(fin>>w)
	{
        function_words.insert(w);
	}

	b = clock();
	cerr<<"loading time: "<<double(b-a)/CLOCKS_PER_SEC<<endl;

	Models models = {src_vocab,tgt_vocab,ruletable,lm_model,NULL,&function_words};
	translate_file(models,para,weight,fns);
	b = clock();
	cerr<<"time cost: "<<double(b-a)/CLOCKS_PER_SEC<<endl;
	return 0;
}
