import csv
import io
import json
import math
import statistics
import urllib.request
from typing import Dict, List

YEARS_FORECAST = 10
TARGET_PRICE_MIN = 1_500_000
TARGET_PRICE_MAX = 1_600_000
OVERBUDGET_ALLOWANCE = 0.10

# Commute constraint: exclude areas generally south of Lake Forest for a Downtown LA commute.
EXCLUDED_SOUTH_OF_LAKE_FOREST = {
    "Aliso Viejo",
    "Coto de Caza",
    "Dana Point",
    "Ladera Ranch",
    "Laguna Beach",
    "Laguna Hills",
    "Laguna Niguel",
    "Laguna Woods",
    "Mission Viejo",
    "Rancho Mission Viejo",
    "Rancho Santa Margarita",
    "San Clemente",
    "San Juan Capistrano",
}


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=90) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def annualize_monthly(series: Dict[str, float]) -> Dict[int, float]:
    by_year: Dict[int, List[float]] = {}
    for ds, val in series.items():
        by_year.setdefault(int(ds[:4]), []).append(val)
    return {y: statistics.fmean(vs) for y, vs in by_year.items()}


def linear_projection(history: Dict[int, float], target_years: List[int]) -> Dict[int, float]:
    xs = sorted(history)
    ys = [history[x] for x in xs]
    if len(xs) < 2:
        return {y: ys[-1] for y in target_years}
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    den = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / den if den else 0.0
    intercept = y_mean - slope * x_mean
    return {y: intercept + slope * y for y in target_years}


def minmax(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def fit_ols(features: List[List[float]], target: List[float]) -> List[float]:
    n, p = len(features), len(features[0])
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for i in range(n):
        xi, yi = features[i], target[i]
        for a in range(p):
            xty[a] += xi[a] * yi
            for b in range(p):
                xtx[a][b] += xi[a] * xi[b]
    aug = [row[:] + [rhs] for row, rhs in zip(xtx, xty)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            continue
        piv = aug[col][col]
        for j in range(col, p + 1):
            aug[col][j] /= piv
        for r in range(p):
            if r != col:
                f = aug[r][col]
                for j in range(col, p + 1):
                    aug[r][j] -= f * aug[col][j]
    return [aug[i][p] for i in range(p)]



def fetch_county_zhvi() -> Dict[int, float]:
    url = "https://files.zillowstatic.com/research/public_csvs/zhvi/County_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
    rows = list(csv.DictReader(io.StringIO(fetch_text(url))))
    row = next(r for r in rows if r.get("RegionName") == "Orange County" and r.get("StateName") == "CA")
    monthly = {}
    for k, v in row.items():
        if k and len(k) >= 7 and k[4] == "-":
            try:
                monthly[k[:7]] = float(v)
            except Exception:
                pass
    return {y: p for y, p in annualize_monthly(monthly).items() if y >= 2012}


def fetch_oc_city_zhvi() -> Dict[str, Dict[int, float]]:
    url = "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
    rows = list(csv.DictReader(io.StringIO(fetch_text(url))))
    out: Dict[str, Dict[int, float]] = {}
    for row in rows:
        if row.get("StateName") != "CA" or row.get("CountyName") != "Orange County":
            continue
        city = row["RegionName"]
        monthly = {}
        for k, v in row.items():
            if k and len(k) >= 7 and k[4] == "-":
                try:
                    monthly[k[:7]] = float(v)
                except Exception:
                    pass
        annual = {y: p for y, p in annualize_monthly(monthly).items() if y >= 2012}
        if len(annual) >= 8:
            out[city] = annual
    return out


def fetch_fred_mortgage_rate() -> Dict[int, float]:
    rows = list(csv.DictReader(io.StringIO(fetch_text("https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"))))
    monthly = {r["observation_date"]: float(r["MORTGAGE30US"]) for r in rows if r["MORTGAGE30US"] != "."}
    return {y: v for y, v in annualize_monthly(monthly).items() if y >= 2012}


def fetch_acs_county(year: int) -> Dict[str, float]:
    vars_ = [
        "B19013_001E", "B01003_001E", "B23025_003E", "B23025_005E",
        "B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
        "B17001_001E", "B17001_002E",
    ]
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5?get={','.join(vars_)}"
        "&for=county:059&in=state:06"
    )
    data = json.loads(fetch_text(url))
    row = dict(zip(data[0], data[1]))
    lf = float(row["B23025_003E"])
    edu_total = float(row["B15003_001E"])
    pov_total = float(row["B17001_001E"])
    bachelors_plus = sum(float(row[v]) for v in ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"])
    return {
        "median_income": float(row["B19013_001E"]),
        "population": float(row["B01003_001E"]),
        "unemployment_rate": float(row["B23025_005E"]) / lf if lf else 0.0,
        "bachelors_plus_share": bachelors_plus / edu_total if edu_total else 0.0,
        "poverty_rate": float(row["B17001_002E"]) / pov_total if pov_total else 0.0,
    }


def fetch_place_acs_2023() -> Dict[str, Dict[str, float]]:
    vars_ = [
        "NAME", "B19013_001E", "B01003_001E", "B23025_003E", "B23025_005E",
        "B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
        "B17001_001E", "B17001_002E",
    ]
    url = f"https://api.census.gov/data/2023/acs/acs5?get={','.join(vars_)}&for=place:*&in=state:06"
    data = json.loads(fetch_text(url))
    header = data[0]
    records = {}
    for r in data[1:]:
        row = dict(zip(header, r))
        name = row["NAME"].replace(" city, California", "").replace(" CDP, California", "")
        try:
            lf = float(row["B23025_003E"])
            edu_total = float(row["B15003_001E"])
            pov_total = float(row["B17001_001E"])
            bachelors_plus = sum(float(row[v]) for v in ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"])
            records[name] = {
                "median_income": float(row["B19013_001E"]),
                "population": float(row["B01003_001E"]),
                "unemployment_rate": float(row["B23025_005E"]) / lf if lf else 0.0,
                "bachelors_plus_share": bachelors_plus / edu_total if edu_total else 0.0,
                "poverty_rate": float(row["B17001_002E"]) / pov_total if pov_total else 0.0,
            }
        except Exception:
            continue
    return records


def cagr(start: float, end: float, years: int) -> float:
    if start <= 0 or end <= 0 or years <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1



def is_commute_eligible(city: str) -> bool:
    return city not in EXCLUDED_SOUTH_OF_LAKE_FOREST

def main() -> None:
    print("Running city-level Orange County model...")
    county_zhvi = fetch_county_zhvi()
    city_zhvi = fetch_oc_city_zhvi()
    mortgage = fetch_fred_mortgage_rate()

    county_acs = {}
    for year in range(2012, 2024):
        try:
            county_acs[year] = fetch_acs_county(year)
        except Exception:
            pass
    place_acs = fetch_place_acs_2023()

    common_years = sorted(set(county_zhvi) & set(mortgage) & set(county_acs))
    X, y = [], []
    for yr in common_years:
        a = county_acs[yr]
        X.append([1.0, float(yr), math.log(a["median_income"]), mortgage[yr], a["unemployment_rate"], math.log(a["population"]), a["bachelors_plus_share"], a["poverty_rate"]])
        y.append(math.log(county_zhvi[yr]))
    beta = fit_ols(X, y)

    last_year = max(common_years)
    forecast_years = [last_year + i for i in range(1, YEARS_FORECAST + 1)]

    income_proj = linear_projection({y: county_acs[y]["median_income"] for y in county_acs}, forecast_years)
    pop_proj = linear_projection({y: county_acs[y]["population"] for y in county_acs}, forecast_years)
    unemp_proj = linear_projection({y: county_acs[y]["unemployment_rate"] for y in county_acs}, forecast_years)
    bach_proj = linear_projection({y: county_acs[y]["bachelors_plus_share"] for y in county_acs}, forecast_years)
    pov_proj = linear_projection({y: county_acs[y]["poverty_rate"] for y in county_acs}, forecast_years)
    mort_proj = linear_projection({y: mortgage[y] for y in common_years}, forecast_years)

    county_prices = {}
    for yr in forecast_years:
        vec = [1.0, float(yr), math.log(max(1.0, income_proj[yr])), mort_proj[yr], max(0.0, unemp_proj[yr]), math.log(max(1.0, pop_proj[yr])), max(0.0, min(1.0, bach_proj[yr])), max(0.0, min(1.0, pov_proj[yr]))]
        county_prices[yr] = math.exp(sum(b * v for b, v in zip(beta, vec)))
    county_growth = cagr(county_prices[forecast_years[0]], county_prices[forecast_years[-1]], YEARS_FORECAST - 1)

    city_rows = []
    sed_inputs = []
    for city, annual in city_zhvi.items():
        if city not in place_acs:
            continue
        latest_year = max(annual)
        start_year = max(min(annual), latest_year - 10)
        hist_growth = cagr(annual[start_year], annual[latest_year], latest_year - start_year)
        blended_growth = 0.60 * hist_growth + 0.40 * county_growth
        blended_growth = max(-0.01, min(0.08, blended_growth))
        p0 = annual[latest_year]
        p10 = p0 * ((1 + blended_growth) ** YEARS_FORECAST)

        a = place_acs[city]
        sed_inputs.append((city, a))
        if not is_commute_eligible(city):
            continue

        city_rows.append({
            "city": city,
            "latest_price": p0,
            "hist_growth_10y": hist_growth,
            "forecast_growth": blended_growth,
            "price_10y": p10,
            "median_income": a["median_income"],
            "bachelors_plus_share": a["bachelors_plus_share"],
            "poverty_rate": a["poverty_rate"],
            "unemployment_rate": a["unemployment_rate"],
        })

    incomes = [r[1]["median_income"] for r in sed_inputs]
    bachelors = [r[1]["bachelors_plus_share"] for r in sed_inputs]
    poverty = [r[1]["poverty_rate"] for r in sed_inputs]
    unemp = [r[1]["unemployment_rate"] for r in sed_inputs]

    for row in city_rows:
        income_s = minmax(row["median_income"], min(incomes), max(incomes))
        bachelors_s = minmax(row["bachelors_plus_share"], min(bachelors), max(bachelors))
        poverty_s = 1 - minmax(row["poverty_rate"], min(poverty), max(poverty))
        unemp_s = 1 - minmax(row["unemployment_rate"], min(unemp), max(unemp))
        sed = 100 * (0.35 * income_s + 0.30 * bachelors_s + 0.20 * poverty_s + 0.15 * unemp_s)
        school = 0.75 * sed + 25 * poverty_s
        row["sed_score"] = max(0.0, min(100.0, sed))
        row["school_quality_proxy_score"] = max(0.0, min(100.0, school))
        row["investment_score"] = 0.50 * row["forecast_growth"] * 100 + 0.35 * row["sed_score"] + 0.15 * row["school_quality_proxy_score"]

    stretch_max = TARGET_PRICE_MAX * (1 + OVERBUDGET_ALLOWANCE)
    midpoint = (TARGET_PRICE_MIN + TARGET_PRICE_MAX) / 2

    strict = [
        r for r in city_rows
        if TARGET_PRICE_MIN <= r["latest_price"] <= stretch_max and r["school_quality_proxy_score"] >= 55
    ]
    for r in strict:
        price_fit = 1 - min(1.0, abs(r["latest_price"] - midpoint) / 250_000)
        r["shop_score"] = 0.55 * r["investment_score"] + 30 * price_fit
    strict = sorted(strict, key=lambda r: r["shop_score"], reverse=True)

    near_band = [
        r for r in city_rows
        if r["latest_price"] <= stretch_max and r["school_quality_proxy_score"] >= 50
    ]
    for r in near_band:
        price_fit = 1 - min(1.0, abs(r["latest_price"] - midpoint) / 300_000)
        r["shop_score"] = 0.45 * r["investment_score"] + 35 * price_fit
    near_band = sorted(near_band, key=lambda r: r["shop_score"], reverse=True)

    seen = set()
    candidates = []
    for pool in (strict, near_band):
        for r in pool:
            if r["city"] in seen:
                continue
            seen.add(r["city"])
            candidates.append(r)
            if len(candidates) >= 5:
                break
        if len(candidates) >= 5:
            break

    recommendations = []
    for r in candidates[:5]:
        in_band = TARGET_PRICE_MIN <= r["latest_price"] <= stretch_max
        recommendations.append({
            "city": r["city"],
            "current_typical_price": round(r["latest_price"], 0),
            "forecast_annual_appreciation": round(r["forecast_growth"] * 100, 2),
            "projected_price_in_10y": round(r["price_10y"], 0),
            "sed_score": round(r["sed_score"], 2),
            "school_quality_proxy_score": round(r["school_quality_proxy_score"], 2),
            "why": "Within target or <=10% over-budget with strong school/prospect balance" if in_band else "Within 10% over-budget allowance and commute-eligible",
        })

    output = {
        "model_version": "v3_city_level_orange_county",
        "target_price_range": [TARGET_PRICE_MIN, TARGET_PRICE_MAX],
        "max_price_with_10pct_stretch": round(stretch_max, 0),
        "commute_filter": "Excluded areas south of Lake Forest",
        "county_price_forecast_10y": {str(y): round(v, 2) for y, v in county_prices.items()},
        "city_count_modeled": len(city_rows),
        "top_city_recommendations": recommendations,
        "all_city_metrics": [
            {
                "city": r["city"],
                "latest_price": round(r["latest_price"], 0),
                "forecast_annual_appreciation_pct": round(r["forecast_growth"] * 100, 2),
                "price_in_10y": round(r["price_10y"], 0),
                "sed_score": round(r["sed_score"], 2),
                "school_quality_proxy_score": round(r["school_quality_proxy_score"], 2),
                "investment_score": round(r["investment_score"], 2),
            }
            for r in sorted(city_rows, key=lambda x: x["investment_score"], reverse=True)
        ],
        "notes": [
            "School projections are socioeconomic proxies (SED + poverty-adjusted school quality proxy) due lack of stable free longitudinal school outcomes API for all Orange County cities.",
            "Use this as a shortlist tool, then validate individual attendance boundaries, school assignment, and property-level comparables before purchase.",
        ],
    }

    with open("orange_county_model_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output["top_city_recommendations"], indent=2))
    print(f"\nModeled {len(city_rows)} cities. Saved orange_county_model_output.json")


if __name__ == "__main__":
    main()
