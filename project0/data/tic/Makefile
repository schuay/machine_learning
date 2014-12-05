all:
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/TicDataDescr.txt
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/dictionary.txt
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tic.data.html
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tic.html
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tic.tar.gz
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tic.task.html
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt
	paste ticeval2000.txt tictgts2000.txt | tr -d '\r' > ticdata2000_f.txt
	cat ticdata2000.txt >> ticdata2000_f.txt
clean:
	rm -f TicDataDescr.txt dictionary.txt tic.data.html tic.html tic.tar.gz tic.task.html ticdata2000.txt ticeval2000.txt tictgts2000.txt ticdata2000_f.txt
