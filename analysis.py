"""Sberbank Russian Housing Market: очистка, EDA, LightGBM, SHAP.

Запуск: python3 analysis.py <папка_с_данными> <папка_результатов>
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3})
BLUE, ORANGE = "#2F6DB5", "#D9822B"
report = {}

# ---------------------------------------------------------------- загрузка
train = pd.read_csv(DATA / "train.csv", parse_dates=["timestamp"])
macro = pd.read_csv(DATA / "macro.csv", parse_dates=["timestamp"])
report["raw_shape"] = list(train.shape)
report["period"] = [str(train.timestamp.min().date()), str(train.timestamp.max().date())]

# пропуски до очистки
na_before = train.isna().mean().sort_values(ascending=False)
na_before[na_before > 0].to_csv(OUT / "missing_before.csv", header=["share_missing"])
report["cols_with_na_before"] = int((na_before > 0).sum())
report["cells_na_before_pct"] = round(float(train.isna().mean().mean() * 100), 2)

fig, ax = plt.subplots(figsize=(8, 6))
top = na_before.head(25)[::-1]
ax.barh(top.index, top.values * 100, color=BLUE)
ax.set_xlabel("% пропусков"); ax.set_title("25 признаков с наибольшей долей пропусков (train)")
fig.tight_layout(); fig.savefig(FIG / "01_missing.png"); plt.close(fig)

# ---------------------------------------------------------------- очистка
df = train.copy()
fixes = {}

def to_nan(mask, col, name):
    fixes[name] = int(mask.sum())
    df.loc[mask, col] = np.nan

# ошибки ввода площадей
to_nan(df.full_sq < 10, "full_sq", "full_sq < 10 м²")
to_nan(df.full_sq > 1000, "full_sq", "full_sq > 1000 м²")
to_nan(df.life_sq > df.full_sq, "life_sq", "life_sq > full_sq")
to_nan(df.life_sq < 5, "life_sq", "life_sq < 5 м²")
to_nan(df.kitch_sq >= df.full_sq, "kitch_sq", "kitch_sq >= full_sq")
to_nan(df.kitch_sq < 2, "kitch_sq", "kitch_sq < 2 м²")
# год постройки: 0, 1, 3, 20052009 и т.п.
to_nan((df.build_year < 1860) | (df.build_year > 2018), "build_year", "build_year вне 1860–2018")
# этажность
to_nan(df.max_floor == 0, "max_floor", "max_floor = 0")
to_nan(df.floor > df.max_floor, "max_floor", "floor > max_floor")
to_nan(df.floor == 0, "floor", "floor = 0")
to_nan(df.num_room == 0, "num_room", "num_room = 0")
# состояние: опечатка 33 → 3
fixes["state = 33 → 3"] = int((df.state == 33).sum())
df.loc[df.state == 33, "state"] = 3
report["fixes"] = fixes

# Подозрительно круглые цены 1/2/3 млн у инвестиционных сделок: это заниженные
# цены в договорах (известная особенность датасета), а не рынок. Убираем их из обучения.
low_price = (df.product_type == "Investment") & (df.price_doc <= 3_000_000) & \
            (df.price_doc.isin([1e6, 2e6, 3e6]))
report["dropped_fake_low_prices"] = int(low_price.sum())
df = df[~low_price].copy()

# Макропоказатели: присоединяем по дате сделки, ffill внутри временного ряда
macro = macro.sort_values("timestamp").ffill()
macro_cols = ["usdrub", "eurrub", "brent", "oil_urals", "cpi", "ppi", "gdp_quart_growth",
              "mortgage_rate", "mortgage_value", "mortgage_growth", "deposits_rate",
              "micex", "micex_rgbi_tr", "rent_price_2room_eco", "rent_price_1room_eco",
              "salary", "unemployment", "average_provision_of_build_contract_moscow"]
macro_cols = [c for c in macro_cols if c in macro.columns]
df = df.merge(macro[["timestamp"] + macro_cols], on="timestamp", how="left")

# инженерные признаки
df["year"] = df.timestamp.dt.year
df["month"] = df.timestamp.dt.month
df["t_months"] = (df.timestamp.dt.year - 2011) * 12 + df.timestamp.dt.month
df["floor_ratio"] = df.floor / df.max_floor
df["life_share"] = df.life_sq / df.full_sq
df["sq_per_room"] = df.full_sq / df.num_room
df["age_at_sale"] = df.year - df.build_year

cat_cols = df.drop(columns=["timestamp"]).select_dtypes(exclude="number").columns.tolist()

# Заполнение пропусков.
# Для числовых: медиана по району (sub_area), иначе глобальная медиана; для признаков
# с большой долей пропусков (>20%) добавляем флаг "_isna", чтобы модель видела сам факт пропуска.
num_cols = [c for c in df.select_dtypes(include="number").columns if c not in ("id", "price_doc")]
na_share = df[num_cols].isna().mean()
for c in na_share[na_share > 0.20].index:
    df[c + "_isna"] = df[c].isna().astype(int)
for c in na_share[na_share > 0].index:
    df[c] = df[c].fillna(df.groupby("sub_area")[c].transform("median")).fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
report["cells_na_after"] = int(df.isna().sum().sum())
df.to_csv(OUT / "train_clean.csv.gz", index=False, compression="gzip")
report["clean_shape"] = list(df.shape)

# ---------------------------------------------------------------- EDA
y = df.price_doc / 1e6
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(y.clip(upper=40), bins=80, color=BLUE); axes[0].set_title("Цена, млн ₽ (обрезано на 40)")
axes[1].hist(np.log1p(df.price_doc), bins=80, color=BLUE); axes[1].set_title("log(1 + цена)")
fig.tight_layout(); fig.savefig(FIG / "02_price_distribution.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
s = df.sample(8000, random_state=0)
for pt, col in [("Investment", BLUE), ("OwnerOccupier", ORANGE)]:
    m = s.product_type == pt
    ax.scatter(s.full_sq[m], s.price_doc[m] / 1e6, s=5, alpha=.35, color=col,
               label={"Investment": "вторичка/инвестиция", "OwnerOccupier": "новостройка (собственник)"}[pt])
ax.set_xlim(0, 200); ax.set_ylim(0, 40); ax.legend()
ax.set_xlabel("Общая площадь, м²"); ax.set_ylabel("Цена, млн ₽"); ax.set_title("Цена и площадь")
fig.tight_layout(); fig.savefig(FIG / "03_price_vs_area.png"); plt.close(fig)

df = df.assign(price_sqm=df.price_doc / df.full_sq)
ts = df.set_index("timestamp").resample("ME").agg(price_sqm=("price_sqm", "median"), n=("id", "size"))
ts = ts.join(macro.set_index("timestamp").resample("ME")["usdrub"].mean())
ts["inv_sqm"] = df[df.product_type == "Investment"].set_index("timestamp").resample("ME")["price_sqm"].median()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ts.index, ts.price_sqm / 1e3, color=BLUE, lw=2, label="все сделки")
ax.plot(ts.index, ts.inv_sqm / 1e3, color=BLUE, lw=1.5, ls="--", label="только вторичка (Investment)")
ax.set_ylabel("тыс. ₽ за м²")
ax2 = ax.twinx(); ax2.plot(ts.index, ts.usdrub, color=ORANGE, lw=1.5, label="USD/RUB"); ax2.set_ylabel("USD/RUB")
ax2.grid(False)
ax.set_title("Медианная цена м² (тыс. ₽) по месяцам и курс доллара")
fig.legend(loc="upper left", bbox_to_anchor=(.08, .9)); fig.tight_layout()
fig.savefig(FIG / "04_price_sqm_time.png"); plt.close(fig)

area = df.groupby("sub_area").agg(price_sqm=("price_sqm", "median"), n=("id", "size"))
area = area[area.n >= 30].sort_values("price_sqm")
pick = pd.concat([area.head(10), area.tail(10)])
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(pick.index, pick.price_sqm / 1e3, color=[ORANGE] * 10 + [BLUE] * 10)
ax.set_xlabel("медиана, тыс. ₽ за м²"); ax.set_title("10 самых дешёвых и 10 самых дорогих районов (≥30 сделок)")
fig.tight_layout(); fig.savefig(FIG / "05_sub_area.png"); plt.close(fig)
report["sub_area_price_sqm_range_k"] = [round(area.price_sqm.min() / 1e3), round(area.price_sqm.max() / 1e3)]

num_now = [c for c in df.select_dtypes(include="number").columns if c not in ("id", "price_doc", "price_sqm")]
corr = df[num_now].corrwith(np.log1p(df.price_doc), method="spearman").dropna()
corr_top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(20)
corr_top.to_csv(OUT / "spearman_top20.csv", header=["spearman_with_log_price"])
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(corr_top.index[::-1], corr_top.values[::-1], color=[BLUE if v > 0 else ORANGE for v in corr_top.values[::-1]])
ax.set_title("Топ-20 признаков по корреляции Спирмена с log(цены)")
fig.tight_layout(); fig.savefig(FIG / "06_correlations.png"); plt.close(fig)

# ---------------------------------------------------------------- модель
features = [c for c in df.columns if c not in ("id", "timestamp", "price_doc", "price_sqm")]
X = df[features].copy()
for c in cat_cols:
    X[c] = X[c].astype("category")
target = np.log1p(df.price_doc)

# Валидация по времени: последние 20% сделок (как в соревновании, где test идёт после train)
order = df.timestamp.argsort().values
cut = int(len(df) * .8)
tr_idx, va_idx = order[:cut], order[cut:]
report["valid_period"] = [str(df.timestamp.iloc[va_idx].min().date()), str(df.timestamp.iloc[va_idx].max().date())]

params = dict(objective="regression", learning_rate=0.03, num_leaves=48, min_child_samples=40,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=42)
dtr = lgb.Dataset(X.iloc[tr_idx], target.iloc[tr_idx])
dva = lgb.Dataset(X.iloc[va_idx], target.iloc[va_idx])
model = lgb.train(params, dtr, 5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
yt = target.iloc[va_idx].values
base = np.full_like(yt, target.iloc[tr_idx].median())
report["model"] = {
    "best_iteration": model.best_iteration,
    "rmsle_valid": round(float(np.sqrt(mean_squared_error(yt, pred))), 4),
    "rmsle_baseline_median": round(float(np.sqrt(mean_squared_error(yt, base))), 4),
    "r2_log_valid": round(float(r2_score(yt, pred)), 4),
    "mape_valid_pct": round(float(mean_absolute_percentage_error(np.expm1(yt), np.expm1(pred)) * 100), 2),
}
# финальная модель на всех данных
final = lgb.train(params, lgb.Dataset(X, target), int(model.best_iteration * 1.1))
final.save_model(str(OUT / "lgbm_model.txt"))

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(np.expm1(yt) / 1e6, np.expm1(pred) / 1e6, s=4, alpha=.3, color=BLUE)
ax.plot([0, 40], [0, 40], color=ORANGE, lw=1)
ax.set_xlim(0, 40); ax.set_ylim(0, 40)
ax.set_xlabel("факт, млн ₽"); ax.set_ylabel("прогноз, млн ₽")
ax.set_title(f"Валидация {report['valid_period'][0]} … {report['valid_period'][1]}")
fig.tight_layout(); fig.savefig(FIG / "07_pred_vs_actual.png"); plt.close(fig)

# ---------------------------------------------------------------- SHAP
# SHAP считаем по финальной модели на выборке из всего периода 2011–2015,
# чтобы макропоказатели оценивались в диапазоне, который модель видела.
Xs = X.sample(5000, random_state=0)
explainer = shap.TreeExplainer(final)
sv = explainer.shap_values(Xs)
imp = pd.Series(np.abs(sv).mean(0), index=features).sort_values(ascending=False)
imp_df = pd.DataFrame({"mean_abs_shap": imp.round(5),
                       "share_pct": (imp / imp.sum() * 100).round(2)})
imp_df.to_csv(OUT / "shap_importance.csv", index_label="feature")
report["shap_top15"] = imp_df.head(15).to_dict(orient="index")

# группы признаков: суммарный вклад
def group(f):
    f0 = f.replace("_isna", "")
    if f0 in ("full_sq", "life_sq", "kitch_sq", "num_room", "life_share", "sq_per_room"):
        return "Площадь и планировка"
    if f0 in ("floor", "max_floor", "floor_ratio", "build_year", "age_at_sale", "material", "state", "product_type"):
        return "Дом, этаж, состояние, тип сделки"
    if f0 in macro_cols or f0 in ("year", "month", "t_months"):
        return "Время и макроэкономика"
    if f0 == "sub_area":
        return "Район (sub_area)"
    if f0.startswith(("cafe_", "office_", "trc_", "leisure_", "sport_", "market_", "church_", "mosque_", "green_part", "prom_part", "big_church")):
        return "Инфраструктура в радиусе 0.5–5 км"
    if f0.endswith(("_km", "_min_walk", "_min_avto", "_avto_min")) or "_km" in f0 or "ID_" in f0:
        return "Расстояния (центр, метро, МКАД и т.д.)"
    return "Характеристики района (население, школы и пр.)"
grp = imp.groupby(imp.index.map(group)).sum().sort_values(ascending=False)
grp_pct = (grp / grp.sum() * 100).round(1)
grp_pct.to_csv(OUT / "shap_importance_by_group.csv", header=["share_pct"], index_label="group")
report["shap_by_group_pct"] = grp_pct.to_dict()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(grp_pct.index[::-1], grp_pct.values[::-1], color=BLUE)
for i, v in enumerate(grp_pct.values[::-1]):
    ax.text(v + .3, i, f"{v:.1f}%", va="center")
ax.set_xlabel("% от суммарного |SHAP|"); ax.set_title("Вклад групп признаков в прогноз цены")
fig.tight_layout(); fig.savefig(FIG / "09b_shap_groups.png"); plt.close(fig)

plt.figure()
shap.summary_plot(sv, Xs, max_display=20, show=False, plot_size=(9, 8))
plt.title("SHAP: влияние признаков на log(цены)"); plt.tight_layout()
plt.savefig(FIG / "08_shap_beeswarm.png"); plt.close("all")

plt.figure()
shap.summary_plot(sv, Xs, plot_type="bar", max_display=20, show=False, plot_size=(9, 7), color=BLUE)
plt.title("SHAP: средний |вклад| (топ-20)"); plt.tight_layout()
plt.savefig(FIG / "09_shap_bar.png"); plt.close("all")

for i, f in enumerate(imp.index[:4]):
    shap.dependence_plot(f, sv, Xs, show=False, interaction_index=None, color=BLUE, alpha=.4)
    plt.gcf().set_size_inches(6, 4); plt.title(f"SHAP dependence: {f}"); plt.tight_layout()
    plt.savefig(FIG / f"10_shap_dep_{i+1}_{f}.png"); plt.close("all")

# эффект в рублях: во сколько раз меняет цену переход от 10-го к 90-му перцентилю признака
eff = {}
for f in imp.index[:10]:
    if f in cat_cols:
        continue
    col = Xs[f].astype(float); s_ = sv[:, features.index(f)]
    lo, hi = col.quantile(.1), col.quantile(.9)
    a = s_[col <= lo].mean(); b = s_[col >= hi].mean()
    eff[f] = {"p10": round(float(lo), 3), "p90": round(float(hi), 3),
              "price_multiplier_p90_vs_p10": round(float(np.exp(b - a)), 3)}
report["effect_p90_vs_p10"] = eff

(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
