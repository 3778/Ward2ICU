.PHONY: README.md

jupyter-kernel:
	poetry run ipython kernel install --user --name=ward2icu

data/mimic:
	mkdir -p data/mimic
	cd data/ && wget -r -N -c -np http://alpha.physionet.org/files/mimiciii-demo/1.4/
data/eicu:
	mkdir -p data/eicu
	cd data/ && wget -r -N -c -np http://alpha.physionet.org/files/eicu-crd-demo/2.0/

data:
	$(MAKE) data/mimic
	$(MAKE) data/eicu

bin/gh-md-toc:
	mkdir -p bin
	wget https://raw.githubusercontent.com/ekalinin/github-markdown-toc/master/gh-md-toc
	chmod a+x gh-md-toc
	mv gh-md-toc bin/

README.md: bin/gh-md-toc
	./bin/gh-md-toc --insert README.md
	rm -f README.md.orig.* README.md.toc.*
