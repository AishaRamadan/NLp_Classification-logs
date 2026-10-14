from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm

def classify(logs)->list:
    labels = []
    for src, log_msg in logs:
        label = classify_log(src,log_msg)
        labels.append(label)

    return labels


def classify_log(src,log_msg):
    label = ''
    if src == 'LegacyCRM':
        label = classify_with_llm(log_msg)
    else:
        label = classify_with_regex(log_msg)
        if not label :
            label = classify_with_bert(log_msg)

    return label


def classify_csv(inputFile):
    import pandas as pd 
    df = pd.read_csv(inputFile)

    df['target label'] = classify(list(zip(df['source'],df['log_message'])))

    df.to_csv('resources/output.csv',index=False)


if __name__ =='__main__':
    classify_csv('resources/test.csv')