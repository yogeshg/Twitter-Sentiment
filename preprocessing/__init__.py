# This Python file uses the following encoding: utf-8
import re

# Hashtags
hash_regex = r"#(\w+)"
def hash_repl(match):
	return 'HASH('+match.group(1).upper()+')'

# Handels
hndl_regex = r"@(\w+)"
def hndl_repl(match):
	return 'HNDL('+match.group(1).upper()+')'

# URLs
url_regex = \
	r"(http|https|ftp)://[a-zA-Z0-9\./]+"

# Emoticons
emoticons = \
	[	('EMOT(SMILEY)',	[':-)', ':)', '(:', '(-:', ] )	,\
		('EMOT(LAUGH)',		[':-D', ':D', 'X-D', 'XD', 'xD', ] )	,\
		('EMOT(LOVE)',		['<3', ':*', ] )	,\
		('EMOT(WINK)',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ] )	,\
		('EMOT(FROWN)',		[':-(', ':(', '(:', '(-:', ] )	,\
		('EMOT(CRY)',		[':,(', ':\'(', ':"(', ':(('] )	,\
	]

# Punctuations
punctuations = \
	[	('PUNC(DOT)',		['.', ] )	,\
		('PUNC(EXCL)',		['!', '¡', ] )	,\
		('PUNC(QUES)',		['?', '¿', ] )	,\
		#FIXME : charachter '...' and regex(?)
		('PUNC(ELLP)',		['...', '…', ] )	,\
		#FIXME : MORE? http://en.wikipedia.org/wiki/Punctuation
	]

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
