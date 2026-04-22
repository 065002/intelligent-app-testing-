import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ─── Configuration ───────────────────────────────────────────────
NUM_RECORDS = 500

APP_VERSIONS   = ["v1.0", "v1.1", "v1.2", "v2.0", "v2.1", "v2.2", "v3.0"]
MODULES        = ["Login", "Payment", "Dashboard", "Search",
                  "Notifications", "Profile", "API", "Database", "UI", "Auth"]
SEVERITIES     = ["Low", "Medium", "High"]
STATUSES       = ["Open", "Fixed", "Reopened"]

BUG_DESCRIPTIONS = [
    "NullPointerException in login flow",
    "Payment gateway timeout",
    "Dashboard not loading on mobile",
    "Search returns wrong results",
    "Notification delay observed",
    "Profile picture not uploading",
    "API returns 500 error",
    "Database connection drops randomly",
    "UI elements misaligned on Safari",
    "Authentication token expiry issue",
    "Crash on empty form submission",
    "Incorrect data displayed after refresh",
    "Memory leak in background service",
    "Slow query execution in reports",
    "Button unresponsive after first click",
    "Session timeout too aggressive",
    "CSV export missing columns",
    "Filter not working on date range",
    "Push notification not received",
    "Password reset link expiring early",
]

# ─── Generate Records ─────────────────────────────────────────────
records = []
start_date = datetime(2023, 1, 1)

for i in range(NUM_RECORDS):
    version      = random.choice(APP_VERSIONS)
    module       = random.choice(MODULES)
    bug_id       = f"BUG-{1000 + i}"
    description  = random.choice(BUG_DESCRIPTIONS)
    severity     = random.choices(SEVERITIES, weights=[0.3, 0.45, 0.25])[0]
    status       = random.choices(STATUSES,   weights=[0.25, 0.55, 0.20])[0]
    occurrences  = random.randint(1, 50)
    time_to_fix  = round(random.uniform(0.5, 30.0), 1)   # days
    release_date = start_date + timedelta(days=random.randint(0, 730))

    # Intentionally introduce ~8% missing values for realism
    if random.random() < 0.08:
        time_to_fix = np.nan
    if random.random() < 0.05:
        severity = np.nan

    records.append([
        version, module, bug_id, description, severity,
        status, occurrences, time_to_fix,
        release_date.strftime("%Y-%m-%d")
    ])

# ─── Build DataFrame ─────────────────────────────────────────────
columns = [
    "App_Version", "Module", "Bug_ID", "Bug_Description",
    "Severity", "Status", "Occurrences", "Time_to_Fix_Days", "Release_Date"
]

df = pd.DataFrame(records, columns=columns)
df.to_csv("bug_dataset.csv", index=False)
print(f"✅ Dataset generated: {len(df)} records → bug_dataset.csv")
print(df.head())
