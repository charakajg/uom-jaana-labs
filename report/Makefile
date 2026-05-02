# Build the project report PDF.
#
#   make            -> compile main.pdf with tectonic and mirror to ../report.pdf
#   make clean      -> remove auxiliary files

TEX      := main.tex
PDF      := main.pdf
ROOT_PDF := ../report.pdf

.PHONY: all clean

all: $(ROOT_PDF)

$(PDF): $(TEX) sections/*.tex references.bib
	tectonic $(TEX)

$(ROOT_PDF): $(PDF)
	cp $(PDF) $(ROOT_PDF)

clean:
	rm -f main.aux main.log main.out main.bbl main.blg main.toc
