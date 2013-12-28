# This Python file uses the following encoding: utf-8
import re

# Hashtags
hash_regex = re.compile(r"#(\w+)")
def hash_repl(match):
	return '__HASH_'+match.group(1).upper()

# Handels
hndl_regex = re.compile(r"@(\w+)")
def hndl_repl(match):
	return '__HNDL_'+match.group(1).upper()

# URLs
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")
# Spliting by word boundaries
word_bound_regex = re.compile(r"\W+")

# Emoticons
emoticons = \
	[	('__EMOT_SMILEY',	[':-)', ':)', '(:', '(-:', ] )	,\
		('__EMOT_LAUGH',		[':-D', ':D', 'X-D', 'XD', 'xD', ] )	,\
		('__EMOT_LOVE',		['<3', ':\*', ] )	,\
		('__EMOT_WINK',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ] )	,\
		('__EMOT_FROWN',		[':-(', ':(', '(:', '(-:', ] )	,\
		('__EMOT_CRY',		[':,(', ':\'(', ':"(', ':(('] )	,\
	]

# Punctuations
punctuations = \
	[	('__PUNC_DOT',		['.', ] )	,\
		('__PUNC_EXCL',		['!', '¡', ] )	,\
		('__PUNC_QUES',		['?', '¿', ] )	,\
		('__PUNC_ELLP',		['...', '…', ] )	,\
		#FIXME : MORE? http://en.wikipedia.org/wiki/Punctuation
	]

#Printing functions for info
def print_config(cfg):
	for (x, arr) in cfg:
		print x, '\t',
		for a in arr:
			print a, '\t',
		print ''

def print_emoticons():
	print_config(emoticons)

def print_punctuations():
	print_config(punctuations)

#For emoticon regexes
def escape_paren(arr):
	return [str.replace(')', '[)}\]]').replace('(', '[({\[]') for str in arr]

def regex_union(arr):
	return '(' + '|'.join( arr ) + ')'

emoticons_regex = [ (repl, re.compile(regex_union(escape_paren(regx))) ) \
					for (repl, regx) in emoticons ]

#For punctuation replacement
def punctuations_repl(match):
	str = match.group(0)
	repl = []
	for (key, parr) in punctuations :
		for punc in parr :
			if punc in str:
				repl.append(key)
	if( len(repl)>0 ) :
		return ' '+' '.join(repl)+' '
	else :
		return str

#FIXME: preprocessing.preprocess()! wtf! will need to move.
def preprocess(str):
	str = re.sub( hash_regex, hash_repl, str )
	str = re.sub( hndl_regex, hndl_repl, str )

	for (repl, regx) in emoticons_regex :
		str = re.sub(regx, ' '+repl+' ', str)
	str = re.sub( word_bound_regex , punctuations_repl, str )

	return str

#from time import time
#import preprocessing, sanderstwitter02
#tweets = sanderstwitter02.getTweetsRawData('sentiment.csv')
#start = time()
#procTweets = [ (preprocessing.preprocess(t),s) for (t,s) in tweets]
#end = time()
#end - start

#uni = [ a if(a[0:2]=='__') else a.lower() for a in re.findall(r"\w+", str) ]
#bi  = nltk.bigrams(uni)
#tri = nltk.bigrams(uni)
