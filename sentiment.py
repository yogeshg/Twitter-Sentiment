"""
Twitter sentiment analysis.

This code performs sentiment analysis on Tweets.

A custom feature extractor looks for key words and emoticons.  These are fed in
to a naive Bayes classifier to assign a label of 'positive', 'negative', or
'neutral'.  Optionally, a principle components transform (PCT) is used to lessen
the influence of covariant features.

"""
import sys, random
import nltk, re
import collections

import time

def get_time_stamp():
    return time.strftime("%y%m%d-%H%M%S-%Z")

TIME_STAMP = get_time_stamp()

NUM_SHOW_FEATURES = 50


def getTrainingAndTestData(tweets, ratio, feature_set):

    add_ngram_feat = feature_set.get('ngram', 1)
    add_negtn_feat = feature_set.get('negtn', False)


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

    train_tweets = tweetsArr[:int(len(tweetsArr)*ratio)]
    test_tweets  = tweetsArr[int(len(tweetsArr)*ratio):]

    unigrams_fd = nltk.FreqDist()
    if add_ngram_feat > 1 :
        n_grams_fd = nltk.FreqDist()

    for( words, sentiment ) in train_tweets:
        words_uni = words
        unigrams_fd.update(words)

        if add_ngram_feat>=2 :
            words_bi  = [ ','.join(map(str,bg)) for bg in nltk.bigrams(words) ]
            n_grams_fd.update( words_bi )

        if add_ngram_feat>=3 :
            words_tri  = [ ','.join(map(str,bg)) for tg in nltk.trigrams(words) ]
            n_grams_fd.update( words_tri )

    sys.stderr.write( '\nlen( unigrams ) = '+str(len( unigrams_fd.keys() )) )

    #unigrams_sorted = nltk.FreqDist(unigrams).keys()
    unigrams_sorted = unigrams_fd.keys()
    #bigrams_sorted = nltk.FreqDist(bigrams).keys()
    #trigrams_sorted = nltk.FreqDist(trigrams).keys()
    if add_ngram_feat > 1 :
        sys.stderr.write( '\nlen( n_grams ) = '+str(len( n_grams_fd )) )
        ngrams_sorted = [ k for (k,v) in n_grams_fd.items() if v>1]
        sys.stderr.write( '\nlen( ngrams_sorted ) = '+str(len( ngrams_sorted )) )

    def get_word_features(words):
        words_uni = words
        words_bi  = [ ','.join(map(str,bg)) for bg in nltk.bigrams(words) ]
        words_tri = [ ','.join(map(str,bg)) for tg in nltk.trigrams(words) ]
        bag = collections.Counter(words_uni+words_bi+words_tri)
        return bag

    negn_regex = re.compile( r"""(?:
        ^(?:never|no|nothing|nowhere|noone|none|not|
            havent|hasnt|hadnt|cant|couldnt|shouldnt|
            wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
        )$
    )
    |
    n't
    """, re.X)

    def get_negation_features(words):
        INF = 0.0
        negn = [ bool(negn_regex.search(w)) for w in words ]
    
        left = [0.0] * len(words)
        prev = 0.0
        for i in range(0,len(words)):
            if( negn[i] ):
                prev = 1.0
            left[i] = prev
            prev = max( 0.0, prev-0.1)
    
        right = [0.0] * len(words)
        prev = 0.0
        for i in reversed(range(0,len(words))):
            if( negn[i] ):
                prev = 1.0
            right[i] = prev
            prev = max( 0.0, prev-0.1)
    
        return dict( zip(
                        ['neg_l('+w+')' for w in  words] + ['neg_r('+w+')' for w in  words],
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
        features = {}

        bag = get_word_features(words)
        for ug in unigrams_sorted:
            features['count(%s)' % str(ug)] = bag[ug]
        
        if add_ngram_feat > 1:
            for ng in ngrams_sorted:
                features['count(%s)' % str(ng)] = bag[ng]
        
        if add_negtn_feat :
            negation_features = get_negation_features(words)
            features.update( negation_features )
 
        sys.stderr.write( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets' )
        return features

    extract_features.count = 0;

    # Apply NLTK's Lazy Map
    v_train = nltk.classify.apply_features(extract_features,train_tweets)
    v_test  = nltk.classify.apply_features(extract_features,test_tweets)

    return (v_train, v_test)

def generateARFF( tweets, fileprefix ):

    (v_train, v_test) = getTrainingAndTestData(tweets,0.9)

    arff_formatter = nltk.classify.weka.ARFF_Formatter.from_train(v_train)

    arff_formatter.write(fileprefix+'_train.arff', v_train)
    arff_formatter.write(fileprefix+'_test.arff', v_test)
    arff_formatter.write(fileprefix+'_all.arff', v_train+v_test)

    return True

def trainAndClassify( tweets, classifier, method, feature_set ):

    INFO = ' '.join( [TIME_STAMP, str(classifier), str(method)] + [ str(k)+':'+str(v) for (k,v) in feature_set.items()] )

    print INFO
    sys.stderr.write( '\n'+ '#'*80 +'\n' + INFO )

    if('NaiveBayesClassifier' == classifier):
        CLASSIFIER = nltk.classify.NaiveBayesClassifier
    elif('MaxentClassifier' == classifier):
        CLASSIFIER = nltk.classify.MaxentClassifier
    elif('SvmClassifier' == classifier):
        CLASSIFIER = nltk.classify.SvmClassifier
    elif('DecisionTreeClassifier' == classifier):
        CLASSIFIER = nltk.classify.DecisionTreeClassifier
        def DecisionTreeClassifier_show_most_informative_features( self, n=10 ):
            text = ''
            for i in range( 1, 10 ):
                text = nltk.classify.DecisionTreeClassifier.pp(self,depth=i)
                if len( text.split('\n') ) > n:
                    break
            print text
        CLASSIFIER.show_most_informative_features = DecisionTreeClassifier_show_most_informative_features

    (v_train, v_test) = getTrainingAndTestData(tweets,0.9, feature_set)

    if '1step' == method:
        sys.stderr.write( '\n[training start]' )
        classifier_tot = CLASSIFIER.train( v_train )
        sys.stderr.flush()
        sys.stderr.write( ' [training complete]' )
        
        print '######################'
        print '1 Step Classifier :', classifier
        accuracy_tot = nltk.classify.accuracy(classifier_tot, v_test)
        print 'Accuracy :', accuracy_tot
        print '######################'
        print classifier_tot.show_most_informative_features(NUM_SHOW_FEATURES)
        print '######################'

        # build confusion matrix over test set
        test_truth   = [s for (t,s) in v_test]
        test_predict = [classifier_tot.classify(t) for (t,s) in v_test]

        print 'Accuracy :', accuracy_tot
        print 'Confusion Matrix'
        print nltk.ConfusionMatrix( test_truth, test_predict )

    elif '2step' == method:
        v_train_obj = [ (text, 'obj') if ((sent=='neg')|(sent=='pos')) else (text, sent) \
                    for (text, sent) in v_train ]
        v_train_sen = [ (text, sent) for (text, sent) in v_train if ((sent=='neg')|(sent=='pos')) ]
        
        v_test_obj  = [ (text, 'obj') if ((sent=='neg')|(sent=='pos')) else (text, sent) \
                    for (text, sent) in v_test ]
        v_test_sen  = [ (text, sent) for (text, sent) in v_test if ((sent=='neg')|(sent=='pos')) ]

        sys.stderr.write( '\n[training start]' )
        classifier_obj = CLASSIFIER.train(v_train_obj)
        sys.stderr.flush()
        sys.stderr.write( ' [training complete]' )

        sys.stderr.write( '\n[training start]' )
        classifier_sen = CLASSIFIER.train(v_train_sen)
        sys.stderr.flush()
        sys.stderr.write( ' [training complete]' )

        print '######################'
        print 'Objectivity Classifier :', classifier
        accuracy_obj = nltk.classify.accuracy(classifier_obj, v_test_obj)
        print 'Accuracy :', accuracy_obj
        print '######################'
        print classifier_obj.show_most_informative_features(NUM_SHOW_FEATURES)
        print '######################'

        test_truth_obj   = [s for (t,s) in v_test_obj]
        test_predict_obj = [classifier_obj.classify(t) for (t,s) in v_test_obj]

        print 'Accuracy :', accuracy_obj
        print 'Confusion Matrix'
        print nltk.ConfusionMatrix( test_truth_obj, test_predict_obj )
        
        print '######################'
        print 'Sentiment Classifier :', classifier
        accuracy_sen = nltk.classify.accuracy(classifier_sen, v_test_sen)
        print 'Accuracy :', accuracy_sen
        print '######################'
        print classifier_sen.show_most_informative_features(NUM_SHOW_FEATURES)
        print '######################'

        test_truth_sen   = [s for (t,s) in v_test_sen]
        test_predict_sen = [classifier_sen.classify(t) for (t,s) in v_test_sen]

        print 'Accuracy :', accuracy_sen
        print 'Confusion Matrix'
        if( len(test_truth_sen) > 0 ):
            print nltk.ConfusionMatrix( test_truth_sen, test_predict_sen )

        v_test2 = [(t,classifier_obj.classify(t)) for (t,s) in v_test_obj]

        test_truth   = [s for (t,s) in v_test]
        test_predict = [classifier_sen.classify(t) if s=='obj' else s for (t,s) in v_test2]

        correct = [ t==p for (t,p) in zip(test_truth, test_predict)]
        accuracy_tot = float(sum(correct))/len(correct) if correct else 0

        print '######################'
        print '2 - Step Classifier :', classifier
        print 'Accuracy :', accuracy_tot
        print 'Confusion Matrix'
        print nltk.ConfusionMatrix( test_truth, test_predict )
        print '######################'

    sys.stderr.write('\nAccuracy : %0.5f\n'%accuracy_tot)
    return None

def main(argv) :
    import sanderstwitter02
    import stanfordcorpus
    import stats

    fileprefix = ''

    if (len(argv) > 0) :
        fileprefix = str(argv[0])

    if( fileprefix=='' ):
        fileprefix = 'logs/data_'+TIME_STAMP
    
    tweets1 = sanderstwitter02.getTweetsRawData('sentiment.csv')
    tweets2 = stanfordcorpus.getNormalisedTweets('stanfordcorpus/'+stanfordcorpus.FULLDATA+'.10000.norm.csv')
    random.shuffle(tweets1)
    random.shuffle(tweets2)
    tweets = tweets1[0:500] + tweets2[0:500]

    #stats.stepStats( tweets, fileprefix )
    #generateARFF(tweets, fileprefix)

    trainAndClassify( tweets, classifier='NaiveBayesClassifier', method='1step', feature_set={'ngram':1, 'negn':False})
    trainAndClassify( tweets, classifier='NaiveBayesClassifier', method='1step', feature_set={'ngram':3, 'negn':False})
    trainAndClassify( tweets, classifier='NaiveBayesClassifier', method='1step', feature_set={'ngram':1, 'negn':True})
    trainAndClassify( tweets, classifier='NaiveBayesClassifier', method='1step', feature_set={'ngram':3, 'negn':True})
    trainAndClassify( tweets, classifier='NaiveBayesClassifier', method='2step', feature_set={'ngram':1, 'negn':False})
    trainAndClassify( tweets, classifier='NaiveBayesClassifier', method='2step', feature_set={'ngram':3, 'negn':False})
    trainAndClassify( tweets, classifier='NaiveBayesClassifier', method='2step', feature_set={'ngram':1, 'negn':True})
    trainAndClassify( tweets, classifier='NaiveBayesClassifier', method='2step', feature_set={'ngram':3, 'negn':True})
    
    sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv[1:])
