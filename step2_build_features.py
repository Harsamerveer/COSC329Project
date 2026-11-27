import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_PATH = f"{DATA_DIR}/features_midterm.csv"
CUTOFF_FRACTION = 0.50  # 50% midpoint of each course

# 1. Load the main tables
print("Loading OULAD tables...")
courses = pd.read_csv(f"{DATA_DIR}/courses.csv", na_values="?")
student_info = pd.read_csv(f"{DATA_DIR}/studentInfo.csv", na_values="?")
student_reg = pd.read_csv(f"{DATA_DIR}/studentRegistration.csv", na_values="?")
assessments = pd.read_csv(f"{DATA_DIR}/assessments.csv", na_values="?")
student_assess = pd.read_csv(f"{DATA_DIR}/studentAssessment.csv", na_values="?")
vle = pd.read_csv(f"{DATA_DIR}/vle.csv", na_values="?")

# Find the midterm cut-off day for each course offering
courses["cutoff_day"] = (courses["module_presentation_length"] * CUTOFF_FRACTION).round().astype(int)
cutoff_lookup = courses[["code_module", "code_presentation", "cutoff_day"]]

# 2. Build the base student table and remove students who already dropped out
df = student_info.merge(
    student_reg,
    on=["code_module", "code_presentation", "id_student"],
    how="left"
).merge(
    cutoff_lookup,
    on=["code_module", "code_presentation"],
    how="left"
)

total_students = len(df)
df = df[df["date_unregistration"].isna() | (df["date_unregistration"] > df["cutoff_day"])].copy()
print(f"Kept {len(df)} of {total_students} students still enrolled past the 50% cut-off.")

# Target: 1 = At-Risk (Fail or Withdrawn), 0 = Not At-Risk (Pass or Distinction)
df["target"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

# 3. Read the large clickstream file in chunks and keep only clicks up to midterm
print("Reading studentVle.csv in chunks...")
vle_dtypes = {
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int32",
}

chunks = []
for chunk in pd.read_csv(f"{DATA_DIR}/studentVle.csv", dtype=vle_dtypes, chunksize=1_000_000):
    chunk = chunk.merge(cutoff_lookup, on=["code_module", "code_presentation"], how="left")
    chunk = chunk[chunk["date"] <= chunk["cutoff_day"]]
    if len(chunk) > 0:
        chunks.append(chunk)

svle = pd.concat(chunks, ignore_index=True)

# Tag forum clicks separately from course material clicks
forum_sites = set(vle.loc[vle["activity_type"] == "forumng", "id_site"])
svle["is_forum"] = svle["id_site"].isin(forum_sites)
svle["is_pre_course"] = svle["date"] < 0
svle["is_last_30d"] = svle["date"] >= (svle["cutoff_day"] - 30)

# 4. Aggregate clickstream engagement per student
print("Aggregating material and forum clicks...")
keys = ["code_module", "code_presentation", "id_student"]

material = svle[~svle["is_forum"]]
forums = svle[svle["is_forum"]]

mat_agg = material.groupby(keys).agg(
    material_clicks=("sum_click", "sum"),
    material_active_days=("date", "nunique"),
    material_unique_sites=("id_site", "nunique"),
).reset_index()

mat_recent = material[material["is_last_30d"]].groupby(keys).agg(
    material_clicks_last_30d=("sum_click", "sum"),
).reset_index()

pre_course = svle[svle["is_pre_course"]].groupby(keys).agg(
    pre_course_clicks=("sum_click", "sum"),
).reset_index()

forum_agg = forums.groupby(keys).agg(
    forum_clicks=("sum_click", "sum"),
    forum_active_days=("date", "nunique"),
).reset_index()

# 5. Aggregate early assessments due and submitted before the midterm cut-off
print("Aggregating early assessments...")
assess_meta = assessments.merge(cutoff_lookup, on=["code_module", "code_presentation"], how="left")
early_assess = assess_meta[
    (assess_meta["assessment_type"] != "Exam")
    & assess_meta["date"].notna()
    & (assess_meta["date"] <= assess_meta["cutoff_day"])
]

due_counts = early_assess.groupby(["code_module", "code_presentation"]).agg(
    early_assessments_due=("id_assessment", "count"),
    early_weight_possible=("weight", "sum"),
).reset_index()

sa = student_assess.merge(
    early_assess[["id_assessment", "code_module", "code_presentation", "date", "weight", "cutoff_day"]],
    on="id_assessment",
    how="inner"
)
sa = sa[sa["date_submitted"] <= sa["cutoff_day"]].copy()
sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
sa["days_late"] = sa["date_submitted"] - sa["date"]
sa["is_late"] = (sa["days_late"] > 0).astype(int)
sa["weighted_points"] = sa["score"].fillna(0) * sa["weight"]

assess_agg = sa.groupby(keys).agg(
    early_assessments_submitted=("id_assessment", "count"),
    mean_early_score=("score", "mean"),
    min_early_score=("score", "min"),
    weighted_points_sum=("weighted_points", "sum"),
    submitted_weight_sum=("weight", "sum"),
    avg_days_late=("days_late", "mean"),
    num_late_submissions=("is_late", "sum"),
).reset_index()

# 6. Merge all feature tables back onto the student list
print("Merging features and encoding columns...")
for table in [mat_agg, mat_recent, pre_course, forum_agg, assess_agg]:
    df = df.merge(table, on=keys, how="left")

df = df.merge(due_counts, on=["code_module", "code_presentation"], how="left")

# Students with no clicks or no submissions get 0 for count columns
zero_cols = [
    "material_clicks", "material_active_days", "material_unique_sites",
    "material_clicks_last_30d", "pre_course_clicks",
    "forum_clicks", "forum_active_days",
    "early_assessments_submitted", "num_late_submissions",
    "weighted_points_sum", "submitted_weight_sum",
    "early_assessments_due", "early_weight_possible",
]
df[zero_cols] = df[zero_cols].fillna(0)

# Ratio features
total_clicks = df["material_clicks"] + df["forum_clicks"]
df["forum_click_ratio"] = np.where(total_clicks > 0, df["forum_clicks"] / total_clicks, 0.0)
df["clicks_per_active_day"] = np.where(
    df["material_active_days"] > 0, df["material_clicks"] / df["material_active_days"], 0.0
)
df["submission_rate"] = np.where(
    df["early_assessments_due"] > 0, df["early_assessments_submitted"] / df["early_assessments_due"], 0.0
)
df["weighted_early_score"] = np.where(
    df["submitted_weight_sum"] > 0, df["weighted_points_sum"] / df["submitted_weight_sum"], 0.0
)

# Keep a flag for missing scores so filling with 0 doesn't confuse "scored 0" with "never submitted"
df["has_early_score"] = df["mean_early_score"].notna().astype(int)
df["mean_early_score"] = df["mean_early_score"].fillna(0.0)
df["min_early_score"] = df["min_early_score"].fillna(0.0)
df["avg_days_late"] = df["avg_days_late"].fillna(0.0)
df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())

# Encode demographic columns into numbers
edu_map = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
imd_map = {
    "0-10": 0, "10-20": 1, "20-30": 2, "30-40": 3, "40-50": 4,
    "50-60": 5, "60-70": 6, "70-80": 7, "80-90": 8, "90-100": 9,
}

df["highest_education_ord"] = df["highest_education"].map(edu_map).fillna(1)
df["age_band_ord"] = df["age_band"].map(age_map).fillna(0)

# Strip "%" because the "10-20" band in OULAD is missing its percent sign
df["imd_band_missing"] = df["imd_band"].isna().astype(int)
df["imd_band_ord"] = df["imd_band"].astype(str).str.replace("%", "", regex=False).map(imd_map)
df["imd_band_ord"] = df["imd_band_ord"].fillna(df["imd_band_ord"].median())

df["gender_M"] = (df["gender"] == "M").astype(int)
df["disability_Y"] = (df["disability"] == "Y").astype(int)

# One-hot encode course code (AAA to GGG)
module_dummies = pd.get_dummies(df["code_module"], prefix="mod", dtype=int)

feature_cols = [
    "material_clicks", "material_active_days", "material_unique_sites",
    "material_clicks_last_30d", "clicks_per_active_day", "pre_course_clicks",
    "forum_clicks", "forum_active_days", "forum_click_ratio",
    "early_assessments_submitted", "submission_rate", "has_early_score",
    "mean_early_score", "min_early_score", "weighted_early_score",
    "avg_days_late", "num_late_submissions",
    "date_registration", "num_of_prev_attempts", "studied_credits",
    "highest_education_ord", "age_band_ord", "imd_band_ord", "imd_band_missing",
    "gender_M", "disability_Y",
]

final_df = pd.concat([df[feature_cols], module_dummies, df[["target"]]], axis=1)
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {OUTPUT_PATH}: {final_df.shape[0]} rows, {final_df.shape[1] - 1} features")
print("Missing values left:", final_df.isna().sum().sum())
print("\nTarget split (0 = Pass/Distinction, 1 = At-Risk):")
print(final_df["target"].value_counts())
print((final_df["target"].value_counts(normalize=True) * 100).round(2))