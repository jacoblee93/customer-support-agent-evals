import os
import shutil
import sqlite3
from datetime import timedelta
import pandas as pd
import requests


db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
db = "travel2.sqlite"
# The backup lets us restart for each tutorial section
backup_file = "travel2.backup.sqlite"
overwrite = False
if overwrite or not os.path.exists(db):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(db, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(db, backup_file)


# Convert the flights to present time for our tutorial
def update_dates(file, now=None):
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    # Use provided now parameter if available, otherwise use current time
    current_time = pd.to_datetime(now) if now else pd.to_datetime("now")
    current_time = current_time.tz_localize(example_time.tz)
    time_diff = current_time - example_time + timedelta(hours=6)

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    return file
