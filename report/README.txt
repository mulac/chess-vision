The final report LaTeX template contains a number of files, many of which currently contain placeholder text that you will need to replace:

finalReport.tex     :	The main file that includes all of the others. Will need to be modified to add new chapters.
config.tex          :   Project title, author, session etc.
prelude.tex         :   List of deliverables.
acknowledge.tex     :	Text for the acknowledgements.
summary.tex         :   Text for the summary that appears near the start of the report.
appendices.tex      :	Text for all of the appendices.
refs.bib            :	Bibtex file for the bibliography.
chapters/...        :	Files for each of the chapters.

There are various ways to edit LaTeX files:
- Online with Overleaf (free subscription for individual accounts at last time of checking).
- Using a dedicated package such as texmaker, texmanager, texshop etc.
- Use you favourite text editor (many source code editors understand LaTeX syntax).

In the latter case you would normally convert the .tex files to PDF from the command line:
> pdflatex finalProject
> bibtex finalProject
> pdflatex finalProject
> pdflatex finalProject
This will produce the file 'finalReport.pdf' with references inserted. If you are not using the .bib bibliograpy file,
you can omit the first two lines.

If you do not have 'pdflatex' installed on your system, you can use 'latex' instead, which produces a .dvi file.
This can then be converted to PDF using 'dvipdf'.

If using a school machine and you get errors about a style file not being recognised, try first loading the most recent
version of texlive (i.e. type 'module avail' to see which is the most recent, and then 'module load texlive/20..').

DAH/13/9/2021
SW/12/3/2015

