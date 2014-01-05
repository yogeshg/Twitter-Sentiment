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

    #FIXME: see from other branch
    tweetsArr = []
    #word_regex = re.compile(r"\w+")
    for (text, sentiment) in procTweets:
        words = [word if(word[0:2]=='__') else word.lower() \
                    for word in text.split() \
                                #re.findall(word_regex, words) \
                    if len(word) >= 3]
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

def main(argv) :
    import sanderstwitter02

    if (len(argv) > 0) :
        sys.stdout = open( str(argv[0]), 'w')
    tweets = sanderstwitter02.getTweetsRawData('sentiment.csv')

    #getPreprocessingStats(tweets)

#    trainAndClassify(tweets, 0)
    trainAndClassify(tweets, 1)
#    trainAndClassify(tweets, 2)
    trainAndClassify(tweets, 3)

    sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv[1:])
