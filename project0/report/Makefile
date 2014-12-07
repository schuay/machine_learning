all: build

build: report slides

report: report.Rnw pqplot.Rnw
	R CMD Sweave $<
	pdflatex -shell-escape report.tex
	pdflatex -shell-escape report.tex

slides: slides.Rnw pqplot.Rnw
	R CMD Sweave $<
	pdflatex -shell-escape slides.tex
	pdflatex -shell-escape slides.tex

clean:
	rm -f {report,slides}.{aux,log,out,tex,sh,exitCode,stderr,stdout,toc}
	rm -f slides.{nav,snm,vrb}
	rm -f *-eps-converted-to.pdf
	rm -f *.eps
	rm -f classifier.tex regressor.tex

dist-clean: clean
	rm -rf {report,slides}.pdf

.PHONY: clean dist-clean
