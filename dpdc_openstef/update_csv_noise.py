import csv
import random
from datetime import datetime, timedelta, timezone
import shutil
import os

def parse_date(date_str):
    # Format: 2023-01-01 06:00:00+00:00
    if '+' in date_str:
        dt_str, tz = date_str.rsplit('+', 1)
        dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M:%S")
        # We treat everything as UTC-like/offset-naive for arithmetic to avoid issues, 
        # then append +00:00 for output string to match format
        return dt
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

def update_data():
    file_path = 'static/master_data_with_forecasted.csv'
    backup_path = 'static/master_data_with_forecasted.csv.bak'
    
    # Create backup
    shutil.copy2(file_path, backup_path)
    
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        print("Error: File is empty")
        return

    last_row = rows[-1]
    last_dt = parse_date(last_row['date_time'])
    
    # End time: previous hour from now
    # Using system local time but stripping tz to match our naive logic above
    # We need to be careful about timezones. The file has +00:00.
    # The system date command returned +06 (Dhaka time).
    # The user implies the CSV timestamps are actually Local Time (Dhaka) labeled as +00:00.
    # So we should generate up to previous hour of Local Time.
    
    # Get current local time
    now_local = datetime.now() # Local time (Dhaka)
    end_dt = now_local.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    
    # Make end_dt naive for comparison with the naive parse of file dates
    end_dt_naive = end_dt.replace(tzinfo=None)
    
    print(f"Last data point: {last_dt}")
    print(f"Target end time (Local/Dhaka): {end_dt_naive}")
    
    if end_dt_naive <= last_dt:
        print("No update needed (current time is not ahead of last data point).")
        return

    # Extract last 1 week (168 hours) as template
    template_len = 168
    if len(rows) < template_len:
        template_rows = rows
    else:
        template_rows = rows[-template_len:]
        
    new_rows = []
    current_dt = last_dt + timedelta(hours=1)
    
    # Variables to add noise to
    noise_cols = ['load', 'forecasted_load', 'temp', 'dwpt', 'rhum', 'wspd', 'pres']
    
    idx = 0
    while current_dt <= end_dt_naive:
        # Get template row (cycling through the last week)
        template_row = template_rows[idx % len(template_rows)]
        
        new_row = template_row.copy()
        new_row['date_time'] = f"{current_dt.strftime('%Y-%m-%d %H:%M:%S')}+00:00"
        new_row['is_holiday'] = '0'
        new_row['holiday_type'] = '0'
        new_row['national_event_type'] = '0'
        
        # Add 1-3% noise
        # Random factor between -3% and +3%? Or 1-3%? 
        # "add a certain noise 1%-3%". I will assume +/- 1% to 3% variation range is meant,
        # or just generic small noise. I will use uniform(0.97, 1.03).
        noise_factor = random.uniform(0.97, 1.03)
        
        for col in noise_cols:
            try:
                val = float(new_row[col])
                # Apply noise
                val_noisy = val * noise_factor
                # Round to appropriate decimals (load is int, temp is 1 decimal)
                if col in ['load', 'forecasted_load']:
                    new_row[col] = str(int(round(val_noisy)))
                else:
                    new_row[col] = f"{val_noisy:.1f}"
            except (ValueError, TypeError):
                pass # Keep original if not parseable
        
        new_rows.append(new_row)
        current_dt += timedelta(hours=1)
        idx += 1
        
    print(f"Generated {len(new_rows)} new rows.")
    
    # Append to file
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_rows)
    
    print("Update complete.")

if __name__ == "__main__":
    update_data()

