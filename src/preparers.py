def convert_inputs_targets_to_messages(
    input_text,
    target_text,
    dset,
    lang
):
    """
    Converts standard [input/output] type rows into the universal messages format.
    """
    return [
        {"from": "user", "text": input_text.strip(), "parent": dset, "langauge": lang},
        {"from": "assistant", "text": target_text.strip(), "parent": 0, "langauge": lang},
    ]

def prepare_bangla_alpaca_orca(row):
    instruction = row.get('instruction', '')
    system_prompt = row.get('system_prompt', '')
    input_text = row.get('input', '')
    input_prompt = system_prompt + instruction + input_text
    language = 'bn'

    output = row.get('output', '')

    return convert_inputs_targets_to_messages(input_prompt.strip(), output.strip(), "bangla_alpaca_orca", language)

def prepare_urdu_instruct_news_article_generation(row):
    input_prompt = row.get('inputs', '')
    output = row.get('targets', '')
    language = 'urd'

    return convert_inputs_targets_to_messages(input_prompt.strip(), output.strip(), "urdu_instruct_news_article_generation", language)


def prepare_urdu_instruct_news_headline_generation(row):
    input_prompt = row.get('inputs', '')
    output = row.get('targets', '')
    language = 'urd'

    return convert_inputs_targets_to_messages(input_prompt.strip(), output.strip(), "urdu_instruct_news_headline_generation", language)


def prepare_urdu_instruct_news_category_classification(row):
    input_prompt = row.get('inputs', '')
    output = row.get('targets', '')
    language = 'urd'

    return convert_inputs_targets_to_messages(input_prompt.strip(), output.strip(), "urdu_instruct_news_category_classification", language)


def prepare_cidar(row):
    input_prompt = row.get('instruction', '')
    output = row.get('ouput', '')
    language = 'ar'

    return convert_inputs_targets_to_messages(input_prompt.strip(), output.strip(), "cidar", language)


def prepare_six_millions_instruction_dataset_for_arabic_llm_ft(row):
    input_text = row.get('input', '')
    instruction = row.get('instruction', '')
    output = row.get('output', '')
    if output is None:
        output = ''

    input_prompt = instruction + input_text
    language = 'ar'
    try:
        return convert_inputs_targets_to_messages(input_prompt.strip(), output.strip(), "six_millions_instruction_dataset_for_arabic_llm_ft", language)
    except AttributeError:
        print(output, type(output))
        # exit(-1)