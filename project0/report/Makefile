all: build

build: report.Rnw
	R CMD Sweave $<
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -rf report.{aux,log,out,tex}

dist-clean: clean
	rm -rf report.pdf

.PHONY: clean dist-clean
