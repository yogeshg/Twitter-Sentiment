"""
Twitter sentiment analysis.

This code performs sentiment analysis on Tweets.

A custom feature extractor looks for key words and emoticons.  These are fed in
to a naive Bayes classifier to assign a label of 'positive', 'negative', or
'neutral'.  Optionally, a principle components transform (PCT) is used to lessen
the influence of covariant features.

"""
import sys, random
import nltk
import collections

import time
TIME_STAMP = time.strftime("%y%m%d-%H%M%S-%Z")




def getTrainingAndTestData2(tweets, ratio):

    from functools import wraps
    import re
    import preprocessing

    procTweets = [ (preprocessing.processAll(text, subject=subj, query=quer), sent)    \
                        for (text, sent, subj, quer) in tweets]

    

    stemmer = nltk.stem.PorterStemmer()

    tweetsArr = []
    for (text, sentiment) in procTweets:
        words = [word if(word[0:2]=='__') else word.lower() \
                    for word in text.split() \
                    if len(word) >= 3]
        words = [stemmer.stem(w) for w in words]
        tweetsArr.append([words, sentiment])

    random.shuffle( tweetsArr )
    train_tweets = tweetsArr[:int(len(tweetsArr)*ratio)]
    test_tweets  = tweetsArr[int(len(tweetsArr)*ratio):]

    unigrams = []
    #bigrams  = []
    #trigrams = []
    #n_grams  = []

    for( words, sentiment ) in train_tweets:
        words_uni = words
        #words_bi  = nltk.bigrams(words)
        #words_tri = nltk.trigrams(words)
        unigrams.extend( words_uni )
        #bigrams.extend(  words_bi  )
        #trigrams.extend( words_tri )
        #n_grams.extend(  words_uni )
        #n_grams.extend(  words_bi  )
        #n_grams.extend(  words_tri )

    unigrams_sorted = nltk.FreqDist(unigrams).keys()
    #bigrams_sorted = nltk.FreqDist(bigrams).keys()
    #trigrams_sorted = nltk.FreqDist(trigrams).keys()
    #ngrams_sorted = nltk.FreqDist(n_grams).keys()

    def word_features(words):
        words_uni = words
        #words_bi  = nltk.bigrams(words)
        #words_tri = nltk.trigrams(words)
        bag = collections.Counter(words_uni)
        #document_words.union(set(words_bi))
        #document_words.union(set(words_tri))
        return bag

    import re

    negn_regex = re.compile( r"""(?:
        ^(?:never|no|nothing|nowhere|noone|none|not|
            havent|hasnt|hadnt|cant|couldnt|shouldnt|
            wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
        )$
    )
    |
    n't
    """, re.X)

    def negation_features(words):
        INF = 10000
        negn = [ bool(negn_regex.search(w)) for w in words ]
    
        left = [INF] * len(words)
        prev = INF
        for i in range(0,len(words)):
            if( negn[i] ):
                prev = 0
            left[i] = prev
            prev = min( INF, prev+1)
    
        right = [INF] * len(words)
        prev = INF
        for i in reversed(range(0,len(words))):
            if( negn[i] ):
                prev = 0
            right[i] = prev
            prev = min( INF, prev+1)
    
        return dict( zip(
                        ['left('+w+')' for w in  words] + ['right('+w+')' for w in  words],
                        left + right ) )
    
    def counter(func):  #http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
        @wraps(func)
        def tmp(*args, **kwargs):
            tmp.count += 1
            return func(*args, **kwargs)
        tmp.count = 0
        return tmp

    @counter    #http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
    def extract_features(words):

        bag = word_features(words)
        negn_features = negation_features(words)

        features = {}
        for word in unigrams_sorted:
            features['count(%s)' % str(word)] = bag[word]
        features.update( negn_features )
 
        sys.stderr.write( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets' )
        return features

    extract_features.count = 0;

    # Apply NLTK's Lazy Map
    v_train = nltk.classify.apply_features(extract_features,train_tweets)
    v_test  = nltk.classify.apply_features(extract_features,test_tweets)

    return (v_train, v_test)

def generateARFF( tweets, filename ):

    (v_train, v_test) = getTrainingAndTestData2(tweets,0.9)

    arff_formatter = nltk.classify.weka.ARFF_Formatter.from_train(v_train)

    arff_formatter.write(filename+'_'+TIME_STAMP+'_train.arff', v_train)
    arff_formatter.write(filename+'_'+TIME_STAMP+'_test.arff', v_test)
    arff_formatter.write(filename+'_'+TIME_STAMP+'_all.arff', v_train+v_test)

    return True

def trainAndClassify( tweets, argument ):

    print '\n', '######################'
    if( argument % 2 == 0):
        print 'features\t: getTrainingAndTestData'
        (v_train, v_test) = getTrainingAndTestData(tweets,0.9)
    else:
        print 'features\t: getTrainingAndTestData2'
        (v_train, v_test) = getTrainingAndTestData2(tweets,0.9)

    # train classifier
    if( (argument/2) % 2 == 0):
        print 'classifier\t: NaiveBayesClassifier'
        classifier = nltk.NaiveBayesClassifier.train(v_train);
    else:
        print 'classifier\t: train_maxent_classifier_with_gis'
        classifier = nltk.classify.maxent.train_maxent_classifier_with_gis(v_train);

    # classify and dump results for interpretation
    accuracy = nltk.classify.accuracy(classifier, v_test)
    print '\n', '######################'
    print 'Accuracy :', accuracy
    print classifier.show_most_informative_features(200)

    # build confusion matrix over test set
    test_truth   = [s for (t,s) in v_test]
    test_predict = [classifier.classify(t) for (t,s) in v_test]

    print '\n', '######################'
    print 'Accuracy :', accuracy
    print '\n', '######################'
    print 'Confusion Matrix'
    print nltk.ConfusionMatrix( test_truth, test_predict )

    return classifier

def trainAndClassify2( tweets, argument ):

    (v_train, v_test) = getTrainingAndTestData2(tweets,0.9)

    v_train_obj = [ (text, 'obj') if ((sent=='neg')|(sent=='pos')) else (text, sent) \
                for (text, sent) in v_train ]
    v_train_sen = [ (text, sent) for (text, sent) in v_train if ((sent=='neg')|(sent=='pos')) ]

    v_test_obj  = [ (text, 'obj') if ((sent=='neg')|(sent=='pos')) else (text, sent) \
                for (text, sent) in v_test ]
    v_test_sen  = [ (text, sent) for (text, sent) in v_test if ((sent=='neg')|(sent=='pos')) ]


    # train classifier
    classifier_obj = nltk.NaiveBayesClassifier.train(v_train_obj);
    classifier_sen = nltk.NaiveBayesClassifier.train(v_train_sen);

    test_truth   = [s for (t,s) in v_test]
    v_test2 = [(t,classifier_obj.classify(t)) for (t,s) in v_test_obj]
    test_predict = [classifier_sen.classify(t) if s=='obj' else s for (t,s) in v_test2]

    correct = [ t==p for (t,p) in zip(test_truth, test_predict)]
    accuracy = float(sum(correct))/len(correct) if correct else 0

    print '\n', '2 - Step Classifier'
    print '\n', '######################'
    print 'Accuracy :', accuracy
    print '\n', '######################'
    print 'Confusion Matrix'
    print nltk.ConfusionMatrix( test_truth, test_predict )

    print '\n', '######################'
    print 'Objectivity Classifier'
    accuracy_obj = nltk.classify.accuracy(classifier_obj, v_test_obj)
    print '\n', '######################'
    print 'Accuracy :', accuracy_obj
    print classifier_obj.show_most_informative_features(200)

    print '\n', '######################'
    print 'Sentiment Classifier'
    accuracy_sen = nltk.classify.accuracy(classifier_sen, v_test_sen)
    print '\n', '######################'
    print 'Accuracy :', accuracy_sen
    print classifier_sen.show_most_informative_features(200)

    test_truth_obj   = [s for (t,s) in v_test_obj]
    test_predict_obj = [classifier_obj.classify(t) for (t,s) in v_test_obj]

    print '\n', '######################'
    print 'Accuracy :', accuracy_obj
    print '\n', '######################'
    print 'Confusion Matrix'
    print nltk.ConfusionMatrix( test_truth_obj, test_predict_obj )

    test_truth_sen   = [s for (t,s) in v_test_sen]
    test_predict_sen = [classifier_sen.classify(t) for (t,s) in v_test_sen]

    print '\n', '######################'
    print 'Accuracy :', accuracy_sen
    print '\n', '######################'
    print 'Confusion Matrix'
    print nltk.ConfusionMatrix( test_truth_sen, test_predict_sen )

    return None

def main(argv) :
    import sanderstwitter02
    import stanfordcorpus

    filename = ''

    if (len(argv) > 0) :
        filename = str(argv[0])
        sys.stdout = open( filename+'_'+TIME_STAMP , 'w')
    
    #tweets = sanderstwitter02.getTweetsRawData('sentiment.csv')
    tweets = stanfordcorpus.getNormalisedTweets('stanfordcorpus/'+stanfordcorpus.FULLDATA+'.norm.csv')

#    preprocessingStats(tweets)
#    trainAndClassify(tweets, 1)
#    trainAndClassify2(tweets, 1)
#    sys.stdout.flush()

    if( filename=='' ):
        filename = 'logs/data'

    generateARFF(tweets, filename)

    sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv[1:])
