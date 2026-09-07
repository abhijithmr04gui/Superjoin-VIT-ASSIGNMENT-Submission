# Starter Datasets

This directory contains two independent datasets for the fact knowledge layer assignment:

- `delhivery/` - three company documents covering corporate, operational, and financial facts across different disclosure formats and dates.
- `india-macroeconomy/` - three institutional reports covering overlapping facts about the Indian economy from different publishers and data vintages.

Each dataset has its own README with document provenance and details of any curated excerpts.

## Why two datasets are included

The pipeline is generic and works on either. The Delhivery set is the
primary dataset used in this README's demo walkthrough (see the root
README's "Demo" section) because its three documents overlap heavily on
FY24 revenue/EBITDA figures reported at different points in time (a 2022
IPO prospectus, the FY24 annual report, and the Q4 FY24 earnings
presentation) - which naturally produces corroboration, contradiction,
and time-based reconciliation cases without any hand-tuning. The
India-macroeconomy set is a second, independent test of generalization:
it was never looked at while building the extraction/comparison prompts
or schema, and running it through the same pipeline with no code changes
is the "new PDF" acceptance check from the assignment (section 41,
step 15-16).
