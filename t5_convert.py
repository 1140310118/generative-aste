import os
import re
from transformers import AutoTokenizer
from utils import load_text, load_json, save_json



def convert(triplet, encoding, words, tokens, sentence):
    aspect, opinion, sentiment = triplet

    def entity_convert(entity):
        start = entity[0]
        end   = entity[-1]+1

        char_start = sum([len(word) for word in words[:start]])
        char_end   = sum([len(word) for word in words[:end]])-1

        token_start = encoding.char_to_token(char_start)
        token_end   = encoding.char_to_token(char_end-1)+1

        # print(start, end, char_start, char_end, token_start, token_end)
        assert token_start is not None

        return [
            token_start,
            token_end,
            tokens[token_start:token_end],
            sentence[char_start:char_end]
        ]

    return {
        'aspect'   : entity_convert(aspect),
        'opinion'  : entity_convert(opinion),
        'sentiment': sentiment
    }



mapping = {
    " .our": "our",
    " .Hanx": ".Hanx",
    " .both": ".both",
    " ..very": "..very",
    " ..all": "..all",
    " ..were": "..were",
    " ...": "...",
    " ..": "..",
    " ,": ",",
    " .": ".",
    " ;" : ";",
    " !" : "!",
    " n't": "n't",
    " 's" : "'s",
    " 'm" : "'m",
    " 're": "'re",
    " 'll": "'ll",
    " 've": "'ve",
    " 'd": "'d",
}


def process(example, tokenizer):
    sentence, triplets = example.split('####')
    triplets = eval(triplets)

    words = sentence.split()
    words = [(mapping[' '+word] if ' '+word in mapping else ' '+word) for word in words]

    if words[0][0] != ' ':
        words[0] = ' ' +words[0]

    for k, v in mapping.items():
        sentence = sentence.replace(k, v)
    sentence = re.sub(r'(?P<v1>[^\d])(?P<v2>\.\d)', lambda x: x.group('v1')+' '+x.group('v2'), sentence)
    sentence = re.sub(r'(?P<v1>\'[sd][^ -\.])', lambda x: ' '+x.group('v1'), sentence)

    encoding = tokenizer(sentence)
    tokens   = tokenizer.tokenize(sentence)

    # print(sentence)
    # print(words)
    assert len(sentence) == sum([len(word) for word in words])-1, '\n' + sentence + '\n' + str(words)

    triplets_token = []
    for triplet in triplets:
        triplet_token = convert(triplet, encoding, words, tokens, sentence)
        triplets_token.append(triplet_token)

    return {
        'sentence': sentence,
        'triplets': triplets_token,
        'tokens'  : str(tokens)
    }




if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_data_dir',    type=str)
    parser.add_argument('--output_data_dir', type=str)
    parser.add_argument('--dataset',         type=str)

    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    for mode in ('train', 'dev', 'test'):

        t5_examples = []
        file_name = os.path.join(args.raw_data_dir, args.dataset, f'{mode}_triplets.txt')
        raw_examples = load_text(file_name)

        for i, example in enumerate(raw_examples):
            if example:
                t5_example = process(example, tokenizer)
                t5_example['ID'] = i
                t5_examples.append(t5_example)

        save_file_name = os.path.join(args.output_data_dir, args.dataset, f'{mode}.json')
        print('save', len(t5_examples), 'to', save_file_name)
        save_json(t5_examples, save_file_name)
