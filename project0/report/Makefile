all: build

build: report.tex
	pdflatex $<
	pdflatex $<

clean:
	rm -rf report.{aux,log,out}

dist-clean: clean
	rm -rf report.pdf

.PHONY: clean dist-clean
