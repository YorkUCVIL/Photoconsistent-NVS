from termcolor import colored

def tlog(s, mode='debug'):
	'''
	prints a string with an appropriate header
	'''
	if mode == 'note':
			header = colored('[Note]','red','on_cyan')
	elif mode == 'iter':
			header = colored('	','grey','on_white')
	elif mode == 'debug':
			header = colored('[Debug]','grey','on_yellow')
	elif mode == 'error':
			header = colored('[Error]','cyan','on_red')
	else:
			header = colored('[Invalid print mode]','white','on_red')

	out = '{} {}'.format(header, s)
	print(out)
