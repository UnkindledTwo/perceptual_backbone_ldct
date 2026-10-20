#!/usr/bin/env bash
# Copies the generated figures, tables and bibliography into paper/, so paper/ compiles on its own (Tectonic or Overleaf).
set -euo pipefail
cd "$(dirname "$0")"
for name in fig1_overview_column fig2_screen_column fig3_qualitative_texture fig4_screen_vs_outcome; do
    cp ../figures/$name.pdf figures/
done
cp ../results/week3/table1_paper_patient.tex ../results/week3/table2_compact.tex tables/
cp ../index.bib index.bib
