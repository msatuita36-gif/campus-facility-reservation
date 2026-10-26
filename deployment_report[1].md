# Short Deployment Report

## Project
Campus Facility Reservation & Decision Support System

## Reused code and materials
The original `midterm.ipynb` was retained in the project package as the original midterm submission. The revised Streamlit app reuses the original project concept: campus facility reservations, deriving a binary reservation outcome, analyzing completion/non-completion patterns, and training a decision-tree classifier. The supplied Excel workbook was converted to `facility_reservations.csv` for reproducible app loading.

## Changes made
- Replaced notebook-only interaction with a browser-based Streamlit interface.
- Implemented five modules: overview dashboard, data explorer, completion prediction, risk/pattern analysis, and decision-support recommendations.
- Used a preprocessing pipeline with imputation and one-hot encoding so categorical values can be handled consistently.
- Used booking-time fields as predictors and excluded post-outcome fields to reduce target leakage.
- Added a stratified 80/20 hold-out evaluation, metrics, confusion matrix, download option, updated requirements, UML use case diagram, and workflow documentation.
- Added warnings that predictions are estimates and human review is required.

## Tests
- Confirmed the supplied workbook can be read and contains 365 records and 13 original columns.
- Checked that the revised project uses the actual workbook column names, including `User_Category`, `Lead_Time_Days`, and `Reservation_Weekday`.
- The app's prediction and dashboard code was prepared for local execution. Before submission, run `streamlit run app.py`, test all five modules, and capture genuine screenshots from the running app.
- A live Streamlit deployment and public GitHub repository have not been created from this package; their URLs must be added after publishing.

## Limitations
The dataset has 365 rows and may not represent future reservation patterns. Cancelled and No-Show records are combined into one `Not Completed` class. The model shows association rather than causation and can make mistakes. The prototype has no login, live database integration, audit log, or automatic production workflow.

## Reflection
This project demonstrates how data mining can support facility planning by summarizing demand, highlighting observed non-completion patterns, and estimating booking outcomes. Building the Streamlit interface makes the analysis easier to use than a notebook alone. The model should remain a decision-support aid: administrators should validate results, monitor outcomes, protect user data, and use human judgment before changing reservation decisions.
