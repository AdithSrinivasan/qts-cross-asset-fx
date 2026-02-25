# QTS Cross-Asset FX Futures Project — TODO (Week 1 Draft)

Objective: Submit a draft including (A) a pitch book and (B) a Jupyter notebook. Performance analysis not required.

## Pitch Book (Slides)
- [x] Create slide deck template
- [ ] Add title slide with project title, student names, and student IDs
- [ ] Add introductory slides motivating the project idea
- [ ] Add hypothesis and trade description slides (lead–lag between equity/credit and FX futures)
- [ ] Add data sources and universe overview (DM/EM selection; FX futures instruments)
- [ ] Add methodology overview (signals, targets, validation plan)
- [ ] Add limitations and risks slide (liquidity, timing, data constraints)
- [ ] Export draft pitch book (PDF/PPTX)

## Jupyter Notebook (Draft)
- [ ] Add notebook header with project title, student names, and student IDs
- [ ] Write detailed project description and motivation
- [ ] Describe hypotheses and economic intuition
  - [ ] Scott: Hypothesis (motivation: equity to FX/futures)
  - [ ] Adith: FX futures > FX spot logic
- [ ] Outline data sources, universe, and sampling frequency
- [ ] Implement initial data ingestion (Bloomberg/DataVento/public sources)
- [ ] Implement initial data cleaning and formatting (dates, joins, missing values)
- [ ] Arrange data into modeling-ready tables (features/targets)
- [ ] Produce at least three exploratory graphs of raw/derived data
  - [ ] Equity index time series (or factor proxy)
  - [ ] FX futures price/return series
  - [ ] Example cross-asset relationship (scatter, rolling correlation, or lead–lag plot)
- [ ] Add brief narrative under each figure describing what is shown

## Repo Setup (Minimum Required)
- [x] Create repository structure (data/, notebooks/, report/)
- [x] Add README.md (project overview + scope for Week 1)
- [x] Add TODO.md (Week 1 scope only)
- [x] Add .gitignore (exclude large raw data and credentials)

## Coordination
- [ ] Finalize country list for initial prototype (5–6 markets)
- [ ] Confirm data availability for selected countries
- [ ] Upload slide deck template to shared drive
- [ ] Commit draft notebook and slides to repository

## Submission Checklist
- [ ] Pitch book includes motivation slides and student names/IDs
- [ ] Notebook includes motivation, hypotheses, data pipeline code, and ≥ 3 graphs
- [ ] All files pushed to repository by Friday night
