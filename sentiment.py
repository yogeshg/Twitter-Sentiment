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

    def counter(func):	#http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
        @wraps(func)
        def tmp(*args, **kwargs):
            tmp.count += 1
            return func(*args, **kwargs)
        tmp.count = 0
        return tmp

    tweetsArr = []
    for (words, sentiment) in tweets:
    	words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    	tweetsArr.append([words_filtered, sentiment])

    random.shuffle( tweetsArr );
    train_tweets = tweetsArr[:int(len(tweetsArr)*ratio)]
    test_tweets  = tweetsArr[int(len(tweetsArr)*ratio):]

    def get_words_in_tweets(tweetsArr):
        all_words = []
        for (words, sentiment) in tweetsArr:
          all_words.extend(words)
        return all_words

    def get_word_features(wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features

    word_features = get_word_features(get_words_in_tweets(train_tweets))

    @counter	#http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        #FIXME why the hell is this line printed after getTrainingAndTestData2 is returned!!!
        sys.stderr.write( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets' )
        return features

    extract_features.count = 0;

    v_train = nltk.classify.apply_features(extract_features,train_tweets)
    v_test  = nltk.classify.apply_features(extract_features,test_tweets)
    #v_train = [(extract_features(train_tweets[i][0]),train_tweets[i][1]) for i in range(len(train_tweets))]
    #v_test  = [(extract_features(test_tweets[i][0]) , test_tweets[i][1]) for i in range(len(test_tweets)) ]
    
    sys.stderr.write('\n')
    sys.stderr.flush()

    #print('returning from ' + getTrainingAndTestData2.__name__);

    return (v_train, v_test)

def trainAndClassify( tweets, argument ):

    print '######################'

    if( argument % 2 == 0):
        print 'features\t: sandersfeatures'
    else:
        print 'features\t: laurentfeatures'

    if( (argument/2) % 2 == 0):
        print 'classifier\t: NaiveBayesClassifier'
    else:
        print 'classifier\t: train_maxent_classifier_with_gis'


    if( argument % 2 == 0):
        (v_train, v_test) = getTrainingAndTestData(tweets,0.9)
    else:
        (v_train, v_test) = getTrainingAndTestData2(tweets,0.9)

    # dump tweets which our feature selector found nothing
    #for i in range(0,len(tweets)):
    #    if tweet_features.is_zero_dict( fvecs[i][0] ):
    #        print tweets[i][1] + ': ' + tweets[i][0]

    # apply PCA reduction
    #(v_train, v_test) = \
    #        tweet_pca.tweet_pca_reduce( v_train, v_test, output_dim=1.0 )

    # train classifier
    
    if( (argument/2) % 2 == 0):
        classifier = nltk.NaiveBayesClassifier.train(v_train);
    else:
        classifier = nltk.classify.maxent.train_maxent_classifier_with_gis(v_train);

    # classify and dump results for interpretation

    accuracy = nltk.classify.accuracy(classifier, v_test)
    print '\nAccuracy %f\n' % accuracy
    print classifier.show_most_informative_features(200)

    # build confusion matrix over test set
    test_truth   = [s for (t,s) in v_test]
    test_predict = [classifier.classify(t) for (t,s) in v_test]

    print 'Confusion Matrix'
    print nltk.ConfusionMatrix( test_truth, test_predict )

    return accuracy

def main() :
    import sanderstwitter02
    tweets = sanderstwitter02.getTweetsRawData('sentiment.csv')

    print trainAndClassify(tweets, 0)
    print trainAndClassify(tweets, 1)
    print trainAndClassify(tweets, 2)
    print trainAndClassify(tweets, 3)

if __name__ == "__main__":
	main()
