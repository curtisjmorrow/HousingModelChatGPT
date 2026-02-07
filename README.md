# Orange County City-Level Home Buying Model (Schools + Appreciation)

This model is tailored to **cities within Orange County, CA** and answers:
1. Which cities are in your **$1.5M–$1.6M** price band today (with up to 10% stretch)?
2. Which of those have stronger **school context** (via SED + school-quality proxy)?
3. Which have stronger **10-year home appreciation prospects**?

## Data sources (free/public)
- Zillow Research: county + city ZHVI home value series
- FRED: 30-year fixed mortgage rate (`MORTGAGE30US`)
- U.S. Census ACS 5-year: county + place-level socioeconomic data

## Model overview

### County appreciation anchor
- Fit transparent OLS on Orange County prices using:
  - income, mortgage rates, unemployment, population, educational attainment, poverty, trend.
- Forecast county 10-year price path.

### City appreciation forecast
- For each Orange County city in Zillow:
  - compute historical city CAGR,
  - blend with county macro-implied growth (60% city history, 40% county macro),
  - project 10-year city value.

### School quality / SED
- Build city-level **SED score (0-100)** from ACS place-level:
  - income (35%), bachelor’s+ share (30%), inverse poverty (20%), inverse unemployment (15%).
- Build **school quality proxy score** from SED and poverty-adjustment.

### Recommendation logic
Cities are shortlisted when:
- they meet the commute filter (**exclude areas south of Lake Forest**),
- typical price is between **$1.5M and $1.6M**, with up to **10% over-budget allowed** (max $1.76M), and
- school-quality proxy score is above threshold.

Then ranked by a combined investment score using appreciation + SED + school proxy.

## Run
```bash
python model_orange_county.py
```

Output:
- `orange_county_model_output.json`

## Important note
This is a practical shortlist model, not investment advice. Before purchase, verify school attendance boundaries and property-level comps.


If fewer than 5 strict matches exist in-band, the model fills remaining slots from near-band commute-eligible cities to still return a top-5 shopping list.
