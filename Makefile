
all:

speechcommands:
    cd experiments/speechcommands && python3 train.py --data_dir=../../data/speech_dataset/ --train_dir=./train --how_many_training_steps=5,5

report: merged.pdf

report.pdf: report.md
	pandoc --bibliography=references.bib -V papersize:a4 -V geometry:margin=1.0in -V fontsize=12pt --toc -Vlof -s report.md -o report.pdf

merged.pdf: report.pdf summary.pdf
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=merged.pdf cover.pdf summary.pdf report.pdf end.pdf

summary.pdf: summary.md
	pandoc -V papersize:a4 -V geometry:margin=1.0in -V fontsize=12pt -s summary.md -o summary.pdf
