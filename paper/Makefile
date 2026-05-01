LATEXMK ?= latexmk
TECTONIC := $(shell command -v tectonic 2>/dev/null)
MAIN := main
RESULTS_FIGS := ../results/figures
RESULTS_TABS := ../results/tables

.PHONY: all paper figures-link tables-link clean view help

help:
	@echo "Targets:"
	@echo "  make paper        - compile main.pdf with latexmk (default)"
	@echo "  make figures-link - symlink ../results/figures into figures/"
	@echo "  make tables-link  - symlink ../results/tables into tables/"
	@echo "  make clean        - remove LaTeX build artefacts"

all: paper

figures-link:
	@mkdir -p figures
	@for f in $(RESULTS_FIGS)/*.pdf; do \
		[ -e "$$f" ] && cp -f $$f figures/ || true; \
	done
	@echo "copied figures from $(RESULTS_FIGS)"

tables-link:
	@mkdir -p tables
	@for f in $(RESULTS_TABS)/*.tsv; do \
		[ -e "$$f" ] && cp -f $$f tables/ || true; \
	done
	@echo "copied tables from $(RESULTS_TABS)"

paper: figures-link tables-link
ifeq ($(TECTONIC),)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
else
	$(TECTONIC) $(MAIN).tex
endif

view: paper
	@open $(MAIN).pdf 2>/dev/null || xdg-open $(MAIN).pdf 2>/dev/null || true

clean:
	-rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.synctex.gz *.toc
	-rm -f sections/*.aux
	-rm -f $(MAIN).pdf
