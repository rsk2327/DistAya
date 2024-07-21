import downloaders
import preparers

MAPPER_DICT = {
    'Bangla_Alpaca_Orca': {
        'download_function': downloaders.download_bangla_alpaca_orca,
        'prepare_function': preparers.prepare_bangla_alpaca_orca
    },
    'Urdu_Instruct_News_Article_Generation': {
        'download_function': downloaders.download_urdu_instruct_news_article_generation,
        'prepare_function': preparers.prepare_urdu_instruct_news_article_generation
    },
    'Urdu_Instruct_News_Headline_Generation':{
        'download_function': downloaders.download_urdu_instruct_news_headline_generation,
        'prepare_function': preparers.prepare_urdu_instruct_news_headline_generation
    },
    'Urdu_Instruct_News_Category_Classification':{
        'download_function': downloaders.download_urdu_instruct_news_category_classification,
        'prepare_function': preparers.prepare_urdu_instruct_news_category_classification
    },
    "cidar":{
        'download_function': downloaders.download_cidar,
        'prepare_function': preparers.prepare_cidar
    },
    "Six_Millions_Instruction_Dataset_For_Arabic_Llm_Ft":{
        'download_function':downloaders.download_six_millions_instruction_dataset_for_arabic_llm_ft,
        'prepare_function': preparers.prepare_six_millions_instruction_dataset_for_arabic_llm_ft
    }
}