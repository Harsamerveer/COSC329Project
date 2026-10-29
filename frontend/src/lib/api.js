import axios from "axios";

export const API_BASE = `${process.env.REACT_APP_BACKEND_URL}/api`;
export const api = axios.create({ baseURL: API_BASE });

export const fmtInt = (n) => (n ?? 0).toLocaleString("en-US");
export const fmt3 = (n) => (n == null ? "—" : Number(n).toFixed(3));
export const pm = (m) => (m ? `${fmt3(m.mean)} ± ${fmt3(m.std)}` : "—");

export const MODEL_COLORS = {
  "Random Forest": "#1E3A8A",
  XGBoost: "#D97706",
  "Logistic Regression": "#0284C7",
  "SVM (linear)": "#166534",
  "Naive Bayes": "#DC2626",
};
export const OUTCOME_COLORS = {
  Pass: "#166534",
  Distinction: "#0284C7",
  Fail: "#DC2626",
  Withdrawn: "#94A3B8",
};
export const SOURCE_COLORS = {
  engagement: "#1E3A8A",
  interaction: "#D97706",
  assessment: "#166534",
};
export const METRICS = [
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
  ["roc_auc", "ROC-AUC"],
];
