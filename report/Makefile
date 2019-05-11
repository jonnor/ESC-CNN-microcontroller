
all: report

report: merged.pdf

pyincludes/%.tex: pyincludes/%.py
	PYTHONPATH=../ python3 $< $@

pyplots/%.png: pyplots/%.py
	PYTHONPATH=../ python3 $< $@

plots/%.tex: plot.py
	PYTHONPATH=../ python3 plot.py $@

plots/%.png: plot.py
	PYTHONPATH=../ python3 plot.py $@


includes: pyincludes/urbansound8k-classes.tex \
    pyincludes/microcontrollers.tex \
    pyincludes/experiment-settings.tex \
    pyincludes/models.tex \
    pyincludes/results.tex \
    pyplots/filterbanks.png \
    pyplots/dataaugmentations.png \
    plots/urbansound8k-examples.png \
    plots/urbansound8k-existing-models-logmel.tex \
    plots/urbansound8k-existing-models-logmel.png


report.pdf: report.md includes
	pandoc --bibliography=references.bib -V documentclass=report -V papersize:a4 -V geometry:margin=1.0in -V fontsize=12pt -H preamble.tex --csl ieee.csl --toc -Vlof -Vlot --pdf-engine-opt=-shell-escape --number-sections -s report.md -o report.pdf

merged.pdf: report.pdf summary.pdf
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=merged.pdf cover.pdf summary.pdf report.pdf end.pdf

summary.pdf: summary.md
	pandoc -V papersize:a4 -V geometry:margin=1.0in -V fontsize=12pt -s summary.md -o summary.pdf

status.pdf: status.md
	pandoc -t beamer -s status.md -o status.pdf --slide-level=2 --mathml
