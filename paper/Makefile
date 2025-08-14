# Makefile for Jörmungandr-Semantica Paper

# Compiler
LATEXMK = latexmk -pdf -pvc -interaction=nonstopmode

# Main document
MAIN = paper

all:
	latexmk -pdf -bibtex -pdflatex="pdflatex -interaction=nonstopmode" $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
	rm -f *.run.xml