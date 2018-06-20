temp-paper.pdf: content/*.tex pictures/RGB-matrix.png pictures/RGB-filter.png surroundings/*
	pdflatex temp-paper
	bibtex temp-paper
	pdflatex temp-paper
	pdflatex temp-paper

pictures/RGB-matrix.png: pictures/RGB-matrix.svg
	inkscape -D -z --file=pictures/RGB-matrix.svg --export-png=pictures/RGB-matrix.png
pictures/RGB-filter.png: pictures/RGB-filter.svg
	inkscape -D -z --file=pictures/RGB-filter.svg --export-png=pictures/RGB-filter.png
	
