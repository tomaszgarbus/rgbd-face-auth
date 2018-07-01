mim_paper.pdf: content/*.tex pictures/RGB-matrix.png pictures/RGB-filter.png pictures/* surroundings/*
	latexmk -pdf mim_paper.tex

.PHONY: clean all

all: mim_paper.pdf

pictures/RGB-matrix.png: pictures/RGB-matrix.svg
	inkscape -D -z --file=pictures/RGB-matrix.svg --export-png=pictures/RGB-matrix.png
pictures/RGB-filter.png: pictures/RGB-filter.svg
	inkscape -D -z --file=pictures/RGB-filter.svg --export-png=pictures/RGB-filter.png

clean:
	rm -f mim_paper.bbl mim_paper.synctex.gz
	rm -f pictures/RGB-filter.png pictures/RGB-matrix.png
	latexmk mim_paper.tex -C
