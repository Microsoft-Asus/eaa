if __name__ == '__main__':
	max_length = 0
	with open('db.vocab', 'r') as f:
		for line in f:
			token = line.split('\t')[0]
			if len(token) > max_length:
				max_length = len(token)
	print max_length
