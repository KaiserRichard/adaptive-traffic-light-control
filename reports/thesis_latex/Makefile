TEX = main

.PHONY: all clean

all:
	xelatex $(TEX).tex
	bibtex $(TEX)
	xelatex $(TEX).tex
	xelatex $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).toc
	rm -f chapters/*.aux

