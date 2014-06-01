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

import time
TIME_STAMP = time.strftime("%y%m%d-%H%M%S-%Z")


def getTrainingAndTestData(tweets, ratio):
    import sandersfeatures
    # tweet_features, tweet_pca
    random.shuffle( tweets );

    #fvecs = nltk.classify.apply_features(sandersfeatures.tweet_features.make_tweet_dict,tweets)
    fvecs = [(sandersfeatures.tweet_features.make_tweet_dict(tweets[i][0]),tweets[i][1]) for i in range(len(tweets))]

    return (fvecs[:int(len(fvecs)*ratio)],fvecs[int(len(fvecs)*ratio):])


def getTrainingAndTestData2(tweets, ratio):

    from functools import wraps
    import re
    import preprocessing

    procTweets = [ (preprocessing.processAll(text, subject=subj, query=quer), sent)    \
                        for (text, sent, subj, quer) in tweets]

    def counter(func):  #http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
        @wraps(func)
        def tmp(*args, **kwargs):
            tmp.count += 1
            return func(*args, **kwargs)
        tmp.count = 0
        return tmp


    stemmer = nltk.stem.PorterStemmer()

    #FIXME: see from other branch
    tweetsArr = []
    #word_regex = re.compile(r"\w+")
    for (text, sentiment) in procTweets:
        words = [word if(word[0:2]=='__') else word.lower() \
                    for word in text.split() \
                                #re.findall(word_regex, words) \
                    if len(word) >= 3]
        words = [stemmer.stem(w) for w in words]
        tweetsArr.append([words, sentiment])

    random.shuffle( tweetsArr )
    train_tweets = tweetsArr[:int(len(tweetsArr)*ratio)]
    test_tweets  = tweetsArr[int(len(tweetsArr)*ratio):]

    #def get_words_in_tweets(tweetsArr):
    #    all_words = []
    #    for (words, sentiment) in tweetsArr:
    #        all_words.extend(words)
    #        all_words.extend(nltk.bigrams(words))
    #        all_words.extend(nltk.trigrams(words))
    #    return all_words
    #def get_word_features(wordlist):
    #    wordlist = nltk.FreqDist(wordlist)
    #    return wordlist

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

    uni_dist = nltk.FreqDist(unigrams)
    #bi_dist  = nltk.FreqDist(bigrams)
    #tri_dist = nltk.FreqDist(trigrams)
    #n_dist   = nltk.FreqDist(n_grams)

    #FIXME : Decide wether or not to choose bi & tri grams!
    #           also chose most "Salient" features!!!

    word_features = uni_dist.keys()
    #word_features = bi_dist.keys()
    #word_features = tri_dist.keys()
    #word_features = n_dist.keys()


    @counter    #http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
    def extract_features(words):
        words_uni = words
        #words_bi  = nltk.bigrams(words)
        #words_tri = nltk.trigrams(words)
        document_words = set(words_uni)
        #document_words.union(set(words_bi))
        #document_words.union(set(words_tri))

        features = {}
        for word in word_features:
            features['contains(%s)' % str(word)] = (word in document_words)
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

def preprocessingStats( tweets ):
    import re
    import preprocessing

    def printStats( tweets, function, filtering=True):
        if( function ):
            procTweets = [ (function(text, subject=subj, query=quer), sent)    \
                            for (text, sent, subj, quer) in tweets]
        else:
            procTweets = [ (text, sent)    \
                            for (text, sent, subj, quer) in tweets]
        tweetsArr = []
        for (text, sentiment) in procTweets:
            words = [word if(word[0:2]=='__') else word.lower() \
                            for word in text.split() \
                            if ( (not filtering) | (len(word) >= 3) ) ]
            tweetsArr.append([words, sentiment])
        # tweetsArr
        unigrams = []
        for (words, sentiment) in tweetsArr:
            unigrams.extend(words)
        # unigram
        message = 'number of words '
        if( function ) :
            message += 'after '+function.__name__+' '
        else :
            message += 'before preprocessing '
        message += ( 'filtered' if filtering else 'not filt' )
        message += '\t'
        # message    
        print message, len(set(unigrams))
        return unigrams

    ###########################################################################  

    print 'for', len(tweets), 'tweets:'

    printStats( tweets, None,                   False   )
    printStats( tweets, None,                   True    )
    printStats( tweets, preprocessing.processHashtags,        True    )
    printStats( tweets, preprocessing.processHandles,         True    )
    printStats( tweets, preprocessing.processUrls,            True    )
    printStats( tweets, preprocessing.processEmoticons,       True    )
    printStats( tweets, preprocessing.processPunctuations,    True    )
    printStats( tweets, preprocessing.processRepeatings,      True    )
    printStats( tweets, preprocessing.processAll,             False   )
    unigrams = \
    printStats( tweets, preprocessing.processAll,             True    )

    #a = set(unigrams)
    #b = set(unigrams)
    #print b.symmetric_difference(a)

    print '###########################################################################'

    n = 0

    num_Handles   =    num_Hashtags  =    num_Urls      =    num_Emoticons = 0
    avg_Handles   =    avg_Hashtags  =    avg_Urls      =    avg_Emoticons = 0.0
    max_Handles   =    max_Hashtags  =    max_Urls      =    max_Emoticons = 0

    cnt_words = cnt_chars = 0
    avg_words = avg_chars = 0.0
    max_words = max_chars = 0

    for (text, sent, subj, quer) in tweets:
        n+=1
        num_Handles   = preprocessing.countHandles(text)
        num_Hashtags  = preprocessing.countHashtags(text)
        num_Urls      = preprocessing.countUrls(text)
        num_Emoticons = preprocessing.countEmoticons(text)

        avg_Handles   = avg_Handles   * (n-1)/n + 1.0*num_Handles   /n
        avg_Hashtags  = avg_Hashtags  * (n-1)/n + 1.0*num_Hashtags  /n
        avg_Urls      = avg_Urls      * (n-1)/n + 1.0*num_Urls      /n
        avg_Emoticons = avg_Emoticons * (n-1)/n + 1.0*num_Emoticons /n

        max_Handles   = max(max_Handles   , num_Handles   )
        max_Hashtags  = max(max_Hashtags  , num_Hashtags  )
        max_Urls      = max(max_Urls      , num_Urls      )
        max_Emoticons = max(max_Emoticons , num_Emoticons )

        cnt_words = len(text.split())
        cnt_chars = len(text)
        avg_words = avg_words*(n-1)/n + cnt_words*1.0/n
        avg_chars = avg_chars*(n-1)/n + cnt_chars*1.0/n
        max_words = max(max_words, cnt_words)
        max_chars = max(max_chars, cnt_chars)

    print 'Feature  ','\t', 'avg'        ,'\t', 'max'        ,'\t', 'of', n, 'tweets'
    print 'Handles  ','\t', avg_Handles  ,'\t', max_Handles  
    print 'Hashtags ','\t', avg_Hashtags ,'\t', max_Hashtags 
    print 'Urls     ','\t', avg_Urls     ,'\t', max_Urls     
    print 'Emoticons','\t', avg_Emoticons,'\t', max_Emoticons

    print 'Words    ','\t', avg_words    ,'\t', max_words
    print 'Chars    ','\t', avg_chars    ,'\t', max_chars

    print '###########################################################################'

    def printFreqDistCSV( dist ):
        print '<FreqDist with', len(dist.keys()), 'samples and', dist._N, 'outcomes>'
        for key in dist.keys():
            print key, '\t,\t', dist[key]
            # can be used to judge entropy of n-grams
            # FIXME: write code if required
            # figure out if we need scanner
            # pos = neg = neu = 0
            # if type is tuple
                # for each tweet
                    # if tweet contains tuple(0)
                        # if next is tuple(1)
                            # if next is tuple(2)
                                # inc pos, neg or neu
            # else type is single word
                # for each tweet
                    # if tweet contains word
                        # inc pos, neg or neu
            # print pos, neg, neu

    #unigrams
    uni_dist = nltk.FreqDist(unigrams)
    print 'Unigrams Distribution'
    printFreqDistCSV(uni_dist)
    uni_dist.plot(50, cumulative=True)

    bigrams = nltk.bigrams(unigrams)
    bi_dist = nltk.FreqDist(bigrams)
    print 'Bigrams Distribution'
    printFreqDistCSV(bi_dist)
    bi_dist.plot(50, cumulative=True)

    trigrams = nltk.trigrams(unigrams)
    tri_dist = nltk.FreqDist(trigrams)
    print 'Trigrams Distribution'
    printFreqDistCSV(tri_dist)
    tri_dist.plot(50, cumulative=True)

def main(argv) :
    import sanderstwitter02

    filename = ''

    if (len(argv) > 0) :
        filename = str(argv[0])
        sys.stdout = open( filename+'_'+TIME_STAMP , 'w')
    tweets = sanderstwitter02.getTweetsRawData('sentiment.csv')

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
