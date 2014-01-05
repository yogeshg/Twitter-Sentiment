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
    return None

def getTrainingAndTestData2(tweets, ratio):
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
    max_Handles   =    max_Hashtags  =    max_Urls      =    max_Emoticons = 0.0

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

    print 'Feature  ','\t', 'avg'        ,'\t', 'max'        ,'\t', 'of', n, 'tweets'
    print 'Handles  ','\t', avg_Handles  ,'\t', max_Handles  
    print 'Hashtags ','\t', avg_Hashtags ,'\t', max_Hashtags 
    print 'Urls     ','\t', avg_Urls     ,'\t', max_Urls     
    print 'Emoticons','\t', avg_Emoticons,'\t', max_Emoticons

    print '###########################################################################'



    uni_dist.plot(50, cumulative=True)


def trainAndClassify( tweets, argument ):
    print '######################'
    return 0

def main(argv) :
    import sanderstwitter02

    if (len(argv) > 0) :
        sys.stdout = open( str(argv[0]), 'w')
    tweets = sanderstwitter02.getTweetsRawData('sentiment.csv')

    preprocessingStats(tweets)

    sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv[1:])
