"""
@package sentiment
Twitter sentiment analysis.

This code performs sentiment analysis on Tweets.

A custom feature extractor looks for key words and emoticons.  These are fed in
to a naive Bayes classifier to assign a label of 'positive', 'negative', or
'neutral'.  Optionally, a principle components transform (PCT) is used to lessen
the influence of covariant features.

"""
import random
import nltk


def getTrainingAndTestData(tweets, ratio):
	import tweet_features, tweet_pca
	random.shuffle( tweets );

	fvecs = nltk.classify.apply_features(tweet_features.make_tweet_dict,tweets)

	return (fvecs[:int(len(fvecs)*ratio)],fvecs[int(len(fvecs)*ratio):])


def getTrainingAndTestData2(tweets, ratio):

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

    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    return (nltk.classify.apply_features(extract_features,train_tweets),
    	nltk.classify.apply_features(extract_features,test_tweets) )

def trainAndClassify( argument ):
	import sanderstwitter02
	tweets = sanderstwitter02.getTweetsRawData('sentiment.csv')
	
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

	print classifier.show_most_informative_features(200)
	accuracy = nltk.classify.accuracy(classifier, v_test)
	print '\nAccuracy %f\n' % accuracy

	# build confusion matrix over test set
	test_truth   = [s for (t,s) in v_test]
	test_predict = [classifier.classify(t) for (t,s) in v_test]

	print 'Confusion Matrix'
	print nltk.ConfusionMatrix( test_truth, test_predict )

	return accuracy

def main() :
	print trainAndClassify(0)
	print trainAndClassify(1)
	print trainAndClassify(2)
	print trainAndClassify(3)

