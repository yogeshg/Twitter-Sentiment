import csv

queryTerms = {\
                'apple'     : ['@apple',    ],  \
                'microsoft' : ['#microsoft', ], \
                'google'    : ['#google', ],    \
                'twitter'   : ['#twitter', ],    \
    }

def getTweetsRawData( fileName ):
    # read all tweets and labels
    fp = open( fileName, 'rb' )
    reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
    tweets = []
    for row in reader:
        tweets.append( [row[4], row[1], row[0], queryTerms[(row[0]).lower()] ] )
    # treat neutral and irrelevant the same
    for t in tweets:
        if (t[1] == 'positive'):
            t[1] = 'pos'
        elif (t[1] == 'negative'):
            t[1] = 'neg'
        elif (t[1] == 'irrelevant')|(t[1] == 'neutral'):
            t[1] = 'neu'

    return tweets # 0: Text # 1: class # 2: subject # 3: query
