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
url_regex = \
	r"(http|https|ftp)://[a-zA-Z0-9\./]+"

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

def escape_paren(arr):
	return [str.replace(')', '[)}\]]').replace('(', '[({\[]') for str in arr]

def regex_union(arr):
	return '(' + '|'.join( arr ) + ')'

emoticons_regex = [ (repl, re.compile(regex_union(escape_paren(regx))) ) \
					for (repl, regx) in emoticons ]

#punctuations_regex = [ (repl, re.compile(regex_union(escape_paren(regx))) ) \
#						for (repl, regx) in punctuations ]
