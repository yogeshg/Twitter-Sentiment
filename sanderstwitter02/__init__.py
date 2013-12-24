import csv

def getTweetsRawData( fileName ):
    # read all tweets and labels
    fp = open( fileName, 'rb' )
    reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
    tweets = []
    for row in reader:
        tweets.append( [row[4], row[1]] );
    # treat neutral and irrelevant the same
    for t in tweets:
        if t[1] == 'irrelevant':
            t[1] = 'neutral'
    return tweets
