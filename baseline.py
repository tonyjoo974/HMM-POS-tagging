"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import Counter

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    cnt_tag = Counter()
    word_dict = {}
    # train
    for i in range(len(train)):
        for j in range(len(train[i])):
            word = train[i][j][0]
            tag = train[i][j][1]
            cnt_tag[tag] += 1
            if word not in word_dict:
                word_dict[word] = {}
            if tag in word_dict[word]:
                word_dict[word][tag] += 1
            else:
                word_dict[word][tag] = 1
    # max_key = max(key for key, value in word_dict[word].items())
    # print(max_key)
    # test
    test_output = []
    for sentence in test:
        wrapper = []
        for word in sentence:
            if word in word_dict:
                tag = max(word_dict[word].keys(), key=(lambda key: word_dict[word][key]))
            else:
                tag = cnt_tag.most_common(1)[0][0]
            wrapper.append((word, tag))
        test_output.append(wrapper)
    return test_output
