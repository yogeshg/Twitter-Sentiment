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

    procTweets = [ (preprocessing.preprocess(t),s) for (t,s) in tweets]
    
    def counter(func):  #http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
        @wraps(func)
        def tmp(*args, **kwargs):
            tmp.count += 1
            return func(*args, **kwargs)
        tmp.count = 0
        return tmp

    tweetsArr = []
    word_regex = re.compile(r"\w+")
    for (words, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                    for word in re.findall(word_regex, words) \
                    if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

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

    @counter    #http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        sys.stderr.write( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets' )
        return features

    extract_features.count = 0;

    # Apply NLTK's Lazy Map
    v_train = nltk.classify.apply_features(extract_features,train_tweets)
    v_test  = nltk.classify.apply_features(extract_features,test_tweets)

    sys.stderr.write('\n')
    sys.stderr.flush()

    return (v_train, v_test)

def getPreprocessingStats( tweets ):
    import re
    import preprocessing

    tweetsArr = []
    for (text, sentiment) in tweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() ]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words before preprocessing :\t',
    print len(set(unigrams))

    ###########################################################################

    tweetsArr = []
    for (text, sentiment) in tweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after filtering :\t',
    print len(set(unigrams))

    ###########################################################################

    procTweets = [ (preprocessing.processHashtags(t),s) for (t,s) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after processHashtags :\t',
    print len(set(unigrams))

    ###########################################################################

    procTweets = [ (preprocessing.processHandles(t),s) for (t,s) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after processHandles :\t',
    print len(set(unigrams))

    ###########################################################################

    procTweets = [ (preprocessing.processUrls(t),s) for (t,s) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after processUrls :\t',
    print len(set(unigrams))

    ###########################################################################

    procTweets = [ (preprocessing.processEmoticons(t),s) for (t,s) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after processEmoticons :\t',
    print len(set(unigrams))

    ###########################################################################

    procTweets = [ (preprocessing.processPunctuations(t),s) for (t,s) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after processPunctuations :\t',
    print len(set(unigrams))

    ###########################################################################

    procTweets = [ (preprocessing.processRepeatings(t),s) for (t,s) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after processRepeatings :\t',
    print len(set(unigrams))

    ###########################################################################

    procTweets = [ (preprocessing.processAll(t),s) for (t,s) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() ]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after processAll :\t',
    print len(set(unigrams))
    a = set(unigrams)

    ###########################################################################

    procTweets = [ (preprocessing.processAll(t),s) for (t,s) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        tweet_uni = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if len(word) >= 3]
        tweetsArr.append([tweet_uni, sentiment])

    unigrams = []
    for (words, sentiment) in tweetsArr:
        unigrams.extend(words)

    print 'number of words after processAll + filtering :\t',
    print len(set(unigrams))
    b = set(unigrams)

    print b.symmetric_difference(a)

    print '###########################################################################'

    n = 0

    num_Handles   = 0
    num_Hashtags  = 0
    num_Urls      = 0
    num_Emoticons = 0

    avg_Handles   = 0.0
    avg_Hashtags  = 0.0
    avg_Urls      = 0.0
    avg_Emoticons = 0.0

    max_Handles   = 0.0
    max_Hashtags  = 0.0
    max_Urls      = 0.0
    max_Emoticons = 0.0

    min_Handles   = 0.0
    min_Hashtags  = 0.0
    min_Urls      = 0.0
    min_Emoticons = 0.0

    for (text, sentiment) in tweets:
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

        min_Handles   = min(min_Handles   , num_Handles   )
        min_Hashtags  = min(min_Hashtags  , num_Hashtags  )
        min_Urls      = min(min_Urls      , num_Urls      )
        min_Emoticons = min(min_Emoticons , num_Emoticons )

    print 'Feature  ','\t', 'min'        ,'\t', 'avg'        ,'\t', 'max'        ,'\t', 'of', n, 'tweets'
    print 'Handles  ','\t', min_Handles  ,'\t', avg_Handles  ,'\t', max_Handles  
    print 'Hashtags ','\t', min_Hashtags ,'\t', avg_Hashtags ,'\t', max_Hashtags 
    print 'Urls     ','\t', min_Urls     ,'\t', avg_Urls     ,'\t', max_Urls     
    print 'Emoticons','\t', min_Emoticons,'\t', avg_Emoticons,'\t', max_Emoticons

    print '###########################################################################'

    uni_dist = nltk.FreqDist(unigrams)

    features = uni_dist.keys()

    for word in features:
        print word, '\t,\t', uni_dist[word]
    

    #uni_dist.plot(50)
    uni_dist.plot(50, cumulative=True)


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

def main(argv) :
    import sanderstwitter02

    if (len(argv) > 0) :
        sys.stdout = open( str(argv[0]), 'w')
    tweets = sanderstwitter02.getTweetsRawData('sentiment.csv')

    getPreprocessingStats(tweets)

#    print trainAndClassify(tweets, 0)
#    print trainAndClassify(tweets, 1)
#    print trainAndClassify(tweets, 2)
#    print trainAndClassify(tweets, 3)

    sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv[1:])
