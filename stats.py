import sys, random
import nltk

import time

import re
import preprocessing

import pylab

def stepStats( tweets, fileprefix, num_bins=10, split='easy' ):
    sizes = [     10000,
                  50000,
                 100000,
                 500000,
                1000000,
                1600000  ]

    tot_size = len(tweets)
    max_digits = len(str(tot_size))

    inc = tot_size/num_bins
    rem = tot_size%num_bins

    if split=='equal':
        sizes = [ int((r+1.0)/num_bins*tot_size) for r in range( num_bins ) ]
    else: # split=='easy'
        sizes = range( 0, tot_size, inc)
        sizes = sizes[1:]+[tot_size]

    for s in sizes:
        preprocessingStats( tweets[0:s], fileprefix+'_%0{0}d'.format(sdig) % s )

def preprocessingStats( tweets, fileprefix ):

    sys.stdout = open( fileprefix+'_stats.txt' , 'w')

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
    pylab
    pylab.show = lambda : pylab.savefig(fileprefix+'_1grams.pdf')
    uni_dist.plot(50, cumulative=True)
    pylab.close()

    bigrams = nltk.bigrams(unigrams)
    bi_dist = nltk.FreqDist(bigrams)
    print 'Bigrams Distribution'
    printFreqDistCSV(bi_dist)
    pylab.show = lambda : pylab.savefig(fileprefix+'_2grams.pdf')
    bi_dist.plot(50, cumulative=True)
    pylab.close()

    trigrams = nltk.trigrams(unigrams)
    tri_dist = nltk.FreqDist(trigrams)
    print 'Trigrams Distribution'
    printFreqDistCSV(tri_dist)
    pylab.show = lambda : pylab.savefig(fileprefix+'_3grams.pdf')
    tri_dist.plot(50, cumulative=True)
    pylab.close()

    pylab.show = lambda : pylab.savefig(fileprefix+'_ngrams.pdf')
    uni_dist.plot(50, cumulative=True)
    bi_dist.plot(50, cumulative=True)
    tri_dist.plot(50, cumulative=True)
    pylab.close()    
