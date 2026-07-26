PAPER := dijkwalk-paper.pdf
BUILD := build
FIGURES := \
	figures/q1-cross-paradigm.pdf \
	figures/q2-phase-diagram.pdf \
	figures/q3-size-generalization.pdf

.PHONY: all clean

all: $(PAPER)

figures/q1-cross-paradigm.pdf: ../paper-results/q1-cross-paradigm.svg
	@mkdir -p figures
	rsvg-convert -f pdf -o $@ $<

figures/q2-phase-diagram.pdf: ../paper-results/q2-phase-diagram.svg
	@mkdir -p figures
	rsvg-convert -f pdf -o $@ $<

figures/q3-size-generalization.pdf: ../paper-results/q3-size-generalization.svg
	@mkdir -p figures
	rsvg-convert -f pdf -o $@ $<

$(PAPER): main.tex references.bib $(FIGURES)
	@mkdir -p $(BUILD)
	tectonic main.tex --outdir $(BUILD) --keep-logs --keep-intermediates
	cp $(BUILD)/main.pdf $@

clean:
	rm -rf $(BUILD) $(PAPER) $(FIGURES)
