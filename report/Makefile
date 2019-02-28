
all: report

report: merged.pdf

pyincludes/%.tex: report/pyincludes/%.py
	PYTHONPATH=experiments/urbansounds python3 $< > $@

includes: pyincludes/urbansound8k-classes.tex

report.pdf: report.md includes
	pandoc --bibliography=references.bib -V papersize:a4 -V geometry:margin=1.0in -V fontsize=12pt -H preamble.tex --toc -Vlof -s report.md -o report.pdf

merged.pdf: report.pdf summary.pdf
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=merged.pdf cover.pdf summary.pdf report.pdf end.pdf

summary.pdf: summary.md
	pandoc -V papersize:a4 -V geometry:margin=1.0in -V fontsize=12pt -s summary.md -o summary.pdf

status.pdf: status.md
	pandoc -t beamer -s status.md -o status.pdf --slide-level=2 --mathml
