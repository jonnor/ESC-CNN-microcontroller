
all: report

report: report.pdf

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
	pandoc --include-before-body=cover.latex --include-before-body=abstract.latex --include-after-body=end.latex --bibliography=references.bib -V papersize:a4 -V geometry:margin=1.0in -V fontsize=12pt -H preamble.tex --csl ieee.csl --toc -Vlof -Vlot --pdf-engine-opt=-shell-escape --number-sections -s report.md -o report.pdf

status.pdf: status.md
	pandoc -t beamer -s status.md -o status.pdf --slide-level=2 --mathml
