SRC = src/stat_dist.c src/linalg.c src/linreg.c src/logreg.c \
      src/lda.c src/classify.c src/linreg_diag.c src/classify_ext.c \
      src/knn.c src/poisson.c src/multilogreg.c

islpstat.so:
	gcc -O2 -shared -fPIC -o islpstat.so $(SRC) -lm -Wall

clean:
	rm -f islpstat.so islpstat.dll

test: islpstat.so
	python3 tests/test_vs_statsmodels.py
