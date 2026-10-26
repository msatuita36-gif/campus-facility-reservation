import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Campus Facility Reservation DSS", page_icon="🏫", layout="wide")
st.title("🏫 Campus Facility Reservation & Decision Support System")
st.caption("Analyze reservation patterns and estimate completion risk using the supplied historical dataset.")

DATA_PATH = Path(__file__).parent / "facility_reservations.csv"
TARGET = "Reservation_Outcome"
CATEGORICAL = ["Facility_Type", "User_Category", "Reservation_Purpose", "Reservation_Weekday"]
NUMERIC = ["Lead_Time_Days", "Participants", "Facility_Capacity", "Duration_Hours", "Start_Hour"]
FEATURES = CATEGORICAL + NUMERIC

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    data["Reservation_Outcome"] = np.where(
        data["Usage_Status"].astype(str).str.strip().str.lower().eq("used"),
        "Completed", "Not Completed"
    )
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in CATEGORICAL:
        data[col] = data[col].astype("string").fillna("Unknown").str.strip()
    return data

@st.cache_resource
def train_model(data):
    X = data[FEATURES].copy()
    y = data[TARGET].copy()
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([("cat", cat_pipe, CATEGORICAL), ("num", num_pipe, NUMERIC)])
    model = Pipeline([("preprocess", pre), ("classifier", DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=42))])
    # Hold-out evaluation is reported separately from the final fit.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    evaluation_model = Pipeline([("preprocess", pre), ("classifier", DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=42))])
    evaluation_model.fit(X_train, y_train)
    pred = evaluation_model.predict(X_test)
    metrics = {"Accuracy": accuracy_score(y_test, pred),
               "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
               "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
               "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0)}
    cm = confusion_matrix(y_test, pred, labels=["Completed", "Not Completed"])
    model.fit(X, y)
    return model, metrics, cm

df = load_data()
model, metrics, cm = train_model(df)

st.sidebar.header("Navigation")
module = st.sidebar.radio("Choose a module", [
    "1. Overview Dashboard",
    "2. Reservation Data Explorer",
    "3. Completion Prediction",
    "4. Risk & Pattern Analysis",
    "5. Decision Support Recommendations"
])
st.sidebar.caption("Predictions are estimates, not guarantees. Confirm decisions with staff.")

if module == "1. Overview Dashboard":
    st.header("Module 1 — Overview Dashboard")
    total = len(df)
    completed = int((df[TARGET] == "Completed").sum())
    not_completed = total - completed
    a,b,c,d = st.columns(4)
    a.metric("Total reservations", f"{total:,}")
    b.metric("Completed", f"{completed:,}")
    c.metric("Cancelled / No-show", f"{not_completed:,}")
    d.metric("Completion rate", f"{completed/total:.1%}" if total else "N/A")
    left,right = st.columns(2)
    with left:
        st.subheader("Reservations by facility")
        counts = df["Facility_Type"].value_counts()
        st.bar_chart(counts)
    with right:
        st.subheader("Observed outcome")
        st.bar_chart(df[TARGET].value_counts())
    st.info("Use observed summaries to understand demand. The prediction module estimates outcome for a proposed reservation.")

elif module == "2. Reservation Data Explorer":
    st.header("Module 2 — Reservation Data Explorer")
    st.write("Filter and inspect the supplied reservation records.")
    f1,f2 = st.columns(2)
    facilities = sorted(df["Facility_Type"].dropna().unique().tolist())
    users = sorted(df["User_Category"].dropna().unique().tolist())
    chosen_fac = f1.multiselect("Facility type", facilities, default=facilities)
    chosen_user = f2.multiselect("User category", users, default=users)
    view = df[df["Facility_Type"].isin(chosen_fac) & df["User_Category"].isin(chosen_user)]
    st.caption(f"Showing {len(view)} of {len(df)} records")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
                       "filtered_reservations.csv", "text/csv")

elif module == "3. Completion Prediction":
    st.header("Module 3 — Reservation Completion Prediction")
    st.write("Enter information known when a reservation is requested. Post-event fields are not used as predictors.")
    with st.form("prediction_form"):
        c1,c2,c3 = st.columns(3)
        facility = c1.selectbox("Facility type", sorted(df["Facility_Type"].dropna().unique()))
        user = c2.selectbox("User category", sorted(df["User_Category"].dropna().unique()))
        purpose = c3.selectbox("Reservation purpose", sorted(df["Reservation_Purpose"].dropna().unique()))
        lead = c1.number_input("Lead time (days)", min_value=0, max_value=365, value=7)
        participants = c2.number_input("Expected participants", min_value=1, max_value=2000, value=30)
        capacity = c3.number_input("Facility capacity", min_value=1, max_value=5000, value=50)
        duration = c1.number_input("Duration (hours)", min_value=1, max_value=24, value=2)
        weekday = c2.selectbox("Reservation weekday (as recorded)", sorted(df["Reservation_Weekday"].dropna().unique(), key=str))
        hour = c3.slider("Start hour (24-hour clock)", 0, 23, 10)
        submit = st.form_submit_button("Predict outcome")
    if submit:
        row = pd.DataFrame([{"Facility_Type":facility, "User_Category":user,
            "Reservation_Purpose":purpose, "Lead_Time_Days":lead, "Participants":participants,
            "Facility_Capacity":capacity, "Duration_Hours":duration,
            "Reservation_Weekday":str(weekday), "Start_Hour":hour}])
        # Ensure categorical values match training representation.
        row["Reservation_Weekday"] = row["Reservation_Weekday"].astype("string")
        probs = model.predict_proba(row[FEATURES])[0]
        classes = model.named_steps["classifier"].classes_
        pred = model.predict(row[FEATURES])[0]
        prob_map = dict(zip(classes, probs))
        st.subheader(f"Predicted outcome: {pred}")
        st.metric("Estimated probability of not completing", f"{prob_map.get('Not Completed',0):.1%}")
        if pred == "Not Completed":
            st.warning("Suggested follow-up: confirm the booking, clarify attendance plans, and review applicable cancellation procedures.")
        else:
            st.success("The model estimates a completed reservation. Continue normal confirmation procedures.")
        st.caption("This is a statistical estimate based on historical records; it should not be the sole basis for denying a reservation.")

elif module == "4. Risk & Pattern Analysis":
    st.header("Module 4 — Risk & Pattern Analysis")
    st.subheader("Observed non-completion rate by user category")
    rate_user = df.assign(Not_Completed=(df[TARGET] == "Not Completed")).groupby("User_Category")["Not_Completed"].mean().sort_values(ascending=False)
    st.bar_chart(rate_user)
    st.dataframe((rate_user*100).round(1).rename("Not completed (%)"), use_container_width=True)
    st.subheader("Observed non-completion rate by lead-time group")
    bins = [-1,2,7,14,np.inf]
    labels = ["Last minute (0–2d)", "Short (3–7d)", "Medium (8–14d)", "Early (15+d)"]
    tmp = df.copy()
    tmp["Lead_Time_Group"] = pd.cut(tmp["Lead_Time_Days"], bins=bins, labels=labels)
    rate_lead = tmp.assign(Not_Completed=(tmp[TARGET] == "Not Completed")).groupby("Lead_Time_Group", observed=False)["Not_Completed"].mean()
    st.bar_chart(rate_lead)
    st.dataframe((rate_lead*100).round(1).rename("Not completed (%)"), use_container_width=True)
    st.subheader("Hold-out model evaluation")
    st.dataframe(pd.DataFrame({"Metric":list(metrics.keys()),"Score":[round(v,3) for v in metrics.values()]}), hide_index=True, use_container_width=True)
    st.write("Confusion matrix (rows = actual; columns = predicted; order: Completed, Not Completed)")
    st.dataframe(pd.DataFrame(cm, index=["Actual Completed","Actual Not Completed"], columns=["Predicted Completed","Predicted Not Completed"]))

elif module == "5. Decision Support Recommendations":
    st.header("Module 5 — Decision Support Recommendations")
    st.write("Recommendations below are rule-based prompts from the observed data, not proof of cause.")
    completion = (df[TARGET] == "Completed").mean()
    rates = df.assign(Not_Completed=(df[TARGET] == "Not Completed")).groupby("Facility_Type")["Not_Completed"].mean().sort_values(ascending=False)
    user_rates = df.assign(Not_Completed=(df[TARGET] == "Not Completed")).groupby("User_Category")["Not_Completed"].mean().sort_values(ascending=False)
    st.metric("Overall observed completion rate", f"{completion:.1%}")
    st.markdown("**Suggested actions for administrators**")
    st.write("1. Review facilities with relatively high observed non-completion rates before changing allocation policies.")
    if len(rates):
        st.write(f"2. Prioritize a review of **{rates.index[0]}** reservations, which has the highest observed non-completion rate in this dataset ({rates.iloc[0]:.1%}).")
    if len(user_rates):
        st.write(f"3. Consider an additional reminder or confirmation check for **{user_rates.index[0]}** bookings ({user_rates.iloc[0]:.1%} observed non-completion). Apply consistently and monitor outcomes.")
    st.write("4. Track cancellation and no-show rates monthly, and compare them with a baseline before judging whether an intervention worked.")
    st.write("5. Validate the model on newer data and consult reservation staff before operational deployment.")
    st.warning("Limitations: dataset is historical and limited to 365 records; outcome labels combine cancellations and no-shows; associations do not establish causation; model performance may change over time.")

with st.expander("About this model and dataset"):
    st.write(f"Rows: {len(df)}. Columns: {len(df.columns)}. Model: decision tree classifier (maximum depth 5).")
    st.write("Target: Completed if Usage_Status is Used; otherwise Not Completed. Predictors are reservation-time fields only. Usage_Status, Actual_Attendance, and Cancellation_Days_Before are excluded from predictors to reduce target leakage.")
