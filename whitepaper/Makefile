temp-paper.pdf: content/*.tex pictures/RGB-matrix.png pictures/RGB-filter.png surroundings/*
	latexmk -pdf temp-paper.tex

pictures/RGB-matrix.png: pictures/RGB-matrix.svg
	inkscape -D -z --file=pictures/RGB-matrix.svg --export-png=pictures/RGB-matrix.png
pictures/RGB-filter.png: pictures/RGB-filter.svg
	inkscape -D -z --file=pictures/RGB-filter.svg --export-png=pictures/RGB-filter.png

.PHONY: clean

clean:
	rm temp-paper.bbl
	latexmk -c
