from datasets import load_dataset

def download_bangla_alpaca_orca():
    return load_dataset('BanglaLLM/bangla-alpaca-orca', split='train')

def download_urdu_instruct_news_article_generation():
    return load_dataset('AhmadMustafa/Urdu-Instruct-News-Article-Generation', split='train')

def download_urdu_instruct_news_headline_generation():
    return load_dataset('AhmadMustafa/Urdu-Instruct-News-Headline-Generation', split='train')

def download_urdu_instruct_news_category_classification():
    return load_dataset('AhmadMustafa/Urdu-Instruct-News-Category-Classification', split='train')

def download_cidar():
    return load_dataset('arbml/CIDAR', split='train')

def download_six_millions_instruction_dataset_for_arabic_llm_ft():
    return load_dataset('akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft', split='train')