import pandas as pd

DATA_DIR = "data"

courses = pd.read_csv(f"{DATA_DIR}/courses.csv")
student_info = pd.read_csv(f"{DATA_DIR}/studentInfo.csv")
student_reg = pd.read_csv(f"{DATA_DIR}/studentRegistration.csv")
vle = pd.read_csv(f"{DATA_DIR}/vle.csv")
assessments = pd.read_csv(f"{DATA_DIR}/assessments.csv")

print("Courses and lengths (days):")
print(courses.head(10))

print("\nFinal results breakdown:")
print(student_info["final_result"].value_counts(dropna=False))

print("\nStudents with an unregistration date:", student_reg["date_unregistration"].notna().sum())

print("\nVLE activity types:")
print(vle["activity_type"].value_counts())

print("\nAssessment types:")
print(assessments["assessment_type"].value_counts())