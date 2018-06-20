all: temp-paper.pdf mim-paper.pdf

.PHONY: clean all

temp-paper.pdf: content/*.tex pictures/RGB-matrix.png pictures/RGB-filter.png surroundings/*
	latexmk -pdf temp-paper.tex

mim-paper.pdf: content/*.tex pictures/RGB-matrix.png pictures/RGB-filter.png surroundings/*
	latexmk -pdf mim-paper.tex

pictures/RGB-matrix.png: pictures/RGB-matrix.svg
	inkscape -D -z --file=pictures/RGB-matrix.svg --export-png=pictures/RGB-matrix.png
pictures/RGB-filter.png: pictures/RGB-filter.svg
	inkscape -D -z --file=pictures/RGB-filter.svg --export-png=pictures/RGB-filter.png

clean:
	rm -f temp-paper.bbl mim-paper.bbl temp-paper.synctex.gz mim-paper.synctex.gz
	rm -f pictures/RGB-filter.png pictures/RGB-matrix.png
	latexmk temp-paper.tex -C
	latexmk mim-paper.tex -C
