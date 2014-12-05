all: build

build: report.Rnw
	R CMD Sweave $<
	pdflatex -shell-escape report.tex
	pdflatex -shell-escape report.tex

clean:
	rm -rf report.{aux,log,out,tex,sh,exitCode,stderr,stdout}
	rm -rf classifier.tex regressor.tex

dist-clean: clean
	rm -rf report.pdf

.PHONY: clean dist-clean
