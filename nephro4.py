import pandas as pd
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import gradio as gr
from itertools import product
import io
import base64
import tempfile
import os
from datetime import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
from ortools.sat.python import cp_model
import random
from deap import base, creator, tools, algorithms
import time

def am_pm(hour):
    """Converts 24-hour time to AM/PM format."""
    period = "AM"
    if hour >= 12:
        period = "PM"
    if hour > 12:
        hour -= 12
    elif hour == 0:
        hour = 12  # Midnight
    return f"{int(hour):02d}:00 {period}"

def show_dataframe(csv_path):
    """Reads a CSV file and returns a Pandas DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        return f"Error loading CSV: {e}"

def optimize_staffing(
    csv_file,
    beds_per_staff,
    max_hours_per_staff,  # This will now be interpreted as hours per 28-day period
    hours_per_cycle,
    rest_days_per_week,
    clinic_start,
    clinic_end,
    overlap_time,
    max_start_time_change,
    exact_staff_count=None,
    overtime_percent=100
):
    # Load data
    try:
        if isinstance(csv_file, str):
            # Handle the case when a filepath is passed directly
            data = pd.read_csv(csv_file)
        elif hasattr(csv_file, 'name'):
            # Handle the case when file object is uploaded through Gradio
            data = pd.read_csv(csv_file.name)
        elif csv_file is None:
            # Create a default DataFrame for testing
            days = range(1, 21)  # 20 days
            data = pd.DataFrame({'day': days})
            # Add 4 cycles per day (5-hour cycles)
            for cycle in range(1, 5):
                data[f'cycle{cycle}'] = 3  # Default 3 beds per cycle
        else:
            # Try direct CSV reading
            data = pd.read_csv(io.StringIO(csv_file.decode('utf-8')))
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        # Create a default DataFrame
        days = range(1, 21)  # 20 days
        data = pd.DataFrame({'day': days})
        # Add 4 cycles per day (5-hour cycles)
        for cycle in range(1, 5):
            data[f'cycle{cycle}'] = 3  # Default 3 beds per cycle
        print("Created default schedule with 20 days and 4 cycles per day")
    
    # Rename the index column if necessary
    if data.columns[0] not in ['day', 'Day', 'DAY']:
        data = data.rename(columns={data.columns[0]: 'day'})
    
    # Fill missing values
    for col in data.columns:
        if col.startswith('cycle'):
            data[col] = data[col].fillna(0)
    
    # Calculate clinic hours
    if clinic_end < clinic_start:
        clinic_hours = 24 - clinic_start + clinic_end
    else:
        clinic_hours = clinic_end - clinic_start
    
    # Get number of days in the dataset
    num_days = len(data)
    
    # Parameters
    BEDS_PER_STAFF = float(beds_per_staff)
    STANDARD_PERIOD_DAYS = 30  # Standard 4-week period
    
    # Scale MAX_HOURS_PER_STAFF based on the ratio of actual days to standard period
    BASE_MAX_HOURS = float(max_hours_per_staff)  # This is for a 28-day period
    MAX_HOURS_PER_STAFF = BASE_MAX_HOURS * (num_days / STANDARD_PERIOD_DAYS)
    
    # Log the adjustment for transparency
    original_results = f"Input max hours per staff (28-day period): {BASE_MAX_HOURS}\n"
    original_results += f"Adjusted max hours for {num_days}-day period: {MAX_HOURS_PER_STAFF:.1f}\n\n"
    
    HOURS_PER_CYCLE = float(hours_per_cycle)
    REST_DAYS_PER_WEEK = int(rest_days_per_week)
    SHIFT_TYPES = [5, 10]  # Modified to match 5-hour cycles
    OVERLAP_TIME = float(overlap_time)
    CLINIC_START = int(clinic_start)
    CLINIC_END = int(clinic_end)
    CLINIC_HOURS = clinic_hours
    MAX_START_TIME_CHANGE = int(max_start_time_change)
    OVERTIME_ALLOWED = 1 + (overtime_percent / 100)  # Convert percentage to multiplier
    
    # Calculate staff needed per cycle (beds/BEDS_PER_STAFF, rounded up)
    for col in data.columns:
        if col.startswith('cycle') and not col.endswith('_staff'):
            data[f'{col}_staff'] = np.ceil(data[col] / BEDS_PER_STAFF)
    
    # Get cycle names and number of cycles
    cycle_cols = [col for col in data.columns if col.startswith('cycle') and not col.endswith('_staff')]
    num_cycles = len(cycle_cols)
    
    # Define cycle times
    cycle_times = {}
    for i, cycle in enumerate(cycle_cols):
        cycle_start = (CLINIC_START + i * HOURS_PER_CYCLE) % 24
        cycle_end = (CLINIC_START + (i + 1) * HOURS_PER_CYCLE) % 24
        cycle_times[cycle] = (cycle_start, cycle_end)
    
    # Get staff requirements
    max_staff_needed = max([data[f'{cycle}_staff'].max() for cycle in cycle_cols])
    
    # Define possible shift start times to align with cycles
    shift_start_times = []
    for i in range(num_cycles):
        cycle_start = (CLINIC_START + i * HOURS_PER_CYCLE) % 24
        shift_start_times.append(cycle_start)
    
    # Generate all possible shifts
    possible_shifts = []
    for duration in SHIFT_TYPES:
        for start_time in shift_start_times:
            end_time = (start_time + duration) % 24
            
            # Create a shift with its coverage of cycles
            shift = {
                'id': f"{int(duration)}hr_{int(start_time):02d}",
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'cycles_covered': set()
            }
            
            # Determine which cycles this shift covers
            for cycle, (cycle_start, cycle_end) in cycle_times.items():
                # Handle overnight cycles
                if cycle_end < cycle_start:  # overnight cycle
                    if start_time >= cycle_start or end_time <= cycle_end or (start_time < end_time and end_time > cycle_start):
                        shift['cycles_covered'].add(cycle)
                else:  # normal cycle
                    shift_end = end_time if end_time > start_time else end_time + 24
                    cycle_end_adj = cycle_end if cycle_end > cycle_start else cycle_end + 24
                    
                    # Check for overlap
                    if not (shift_end <= cycle_start or start_time >= cycle_end_adj):
                        shift['cycles_covered'].add(cycle)
                            
            if shift['cycles_covered']:  # Only add shifts that cover at least one cycle
                possible_shifts.append(shift)
    
    # Estimate minimum number of staff needed - more precise calculation
    total_staff_hours = 0
    for _, row in data.iterrows():
        for cycle in cycle_cols:
            total_staff_hours += row[f'{cycle}_staff'] * HOURS_PER_CYCLE
    
    # Calculate theoretical minimum staff with perfect utilization
    theoretical_min_staff = np.ceil(total_staff_hours / MAX_HOURS_PER_STAFF)
    
    # Add a small buffer for rest day constraints
    min_staff_estimate = np.ceil(theoretical_min_staff * (7 / (7 - REST_DAYS_PER_WEEK)))
    
    # Use exact_staff_count if provided, otherwise estimate
    if exact_staff_count is not None and exact_staff_count > 0:
        # When exact staff count is provided, only create that many staff in the model
        estimated_staff = exact_staff_count
        num_staff_to_create = exact_staff_count  # Only create exactly this many staff
    else:
        # Add some buffer for constraints like rest days and shift changes
        estimated_staff = max(min_staff_estimate, max_staff_needed + 1)
        num_staff_to_create = int(estimated_staff)  # Create the estimated number of staff
    
    def optimize_schedule(num_staff, time_limit=600):
        try:
            # Create a binary linear programming model
            model = pl.LpProblem("Staff_Scheduling", pl.LpMinimize)
            
            # Decision variables
            x = pl.LpVariable.dicts("shift", 
                                   [(s, d, shift['id']) for s in range(1, num_staff+1) 
                                                        for d in range(1, num_days+1) 
                                                        for shift in possible_shifts],
                                   cat='Binary')
            
            # Staff usage variable (1 if staff s is used at all, 0 otherwise)
            staff_used = pl.LpVariable.dicts("staff_used", range(1, num_staff+1), cat='Binary')
            
            # Variables for constraint violations
            monthly_hours_violation = pl.LpVariable("monthly_hours_violation", lowBound=0)
            overlap_violation = pl.LpVariable("overlap_violation", lowBound=0)
            timing_violation = pl.LpVariable("timing_violation", lowBound=0)
            coverage_violation = pl.LpVariable("coverage_violation", lowBound=0)
            overtime_violation = pl.LpVariable("overtime_violation", lowBound=0)
            
            # Objective function with numerical priorities
            model += (
                10 * monthly_hours_violation +     # Priority 1: Monthly hours
                8 * overlap_violation +            # Priority 2: Overlap requirements
                6 * timing_violation +             # Priority 3: Week timing consistency
                4 * coverage_violation +           # Priority 4: Coverage satisfaction
                2 * overtime_violation +           # Priority 5: Zero overtime
                1 * pl.lpSum(staff_used[s] for s in range(1, num_staff+1))  # Priority 6: Minimize staff count
            )
            
            # Priority 1: Monthly Hours Constraint
            for s in range(1, num_staff+1):
                staff_hours = pl.lpSum(x[(s, d, shift['id'])] * shift['duration'] 
                             for d in range(1, num_days+1) 
                             for shift in possible_shifts)
                model += staff_hours - MAX_HOURS_PER_STAFF <= monthly_hours_violation
            
            # Priority 2: Overlap Requirements
            for d in range(1, num_days+1):
                for cycle in cycle_cols:
                    overlap_staff = pl.lpSum(x[(s, d, shift['id'])] 
                                 for s in range(1, num_staff+1)
                                 for shift in possible_shifts
                                 if any(c in shift['cycles_covered'] for c in cycle_cols))
                    model += OVERLAP_TIME - overlap_staff <= overlap_violation
            
            # Priority 3: Week Timing Consistency (±1 hour)
            for s in range(1, num_staff+1):
                for w in range((num_days + 6) // 7):
                    week_start = w*7 + 1
                    week_end = min(week_start + 6, num_days)
                    
                    # Reference shift time for the week
                    ref_time = pl.LpVariable(f"ref_time_{s}_{w}", lowBound=0, upBound=23)
                    
                    for d in range(week_start, week_end+1):
                        for shift in possible_shifts:
                            time_diff = pl.LpVariable(f"time_diff_{s}_{d}_{shift['id']}", lowBound=0)
                            model += time_diff >= shift['start'] - ref_time - 1
                            model += time_diff >= ref_time - shift['start'] - 1
                            model += time_diff <= timing_violation
            
            # Priority 4: Coverage Requirements
            for d in range(1, num_days+1):
                day_index = d - 1
                for cycle in cycle_cols:
                    staff_needed = data.iloc[day_index][f'{cycle}_staff']
                    covering_shifts = [shift for shift in possible_shifts if cycle in shift['cycles_covered']]
                    assigned_staff = pl.lpSum(x[(s, d, shift['id'])] 
                                 for s in range(1, num_staff+1) 
                                 for shift in covering_shifts)
                    model += staff_needed - assigned_staff <= coverage_violation
            
            # Priority 5: Zero Overtime
            for s in range(1, num_staff+1):
                weekly_hours = pl.lpSum(x[(s, d, shift['id'])] * shift['duration'] 
                             for d in range(1, num_days+1) 
                             for shift in possible_shifts)
                model += weekly_hours - (40 * (num_days / 7)) <= overtime_violation
            
            # Basic feasibility constraints
            # Each staff works at most one shift per day
            for s in range(1, num_staff+1):
                for d in range(1, num_days+1):
                    model += pl.lpSum(x[(s, d, shift['id'])] for shift in possible_shifts) <= 1
            
            # Link staff_used variable
            for s in range(1, num_staff+1):
                model += pl.lpSum(x[(s, d, shift['id'])] 
                                 for d in range(1, num_days+1) 
                                 for shift in possible_shifts) <= num_days * staff_used[s]
            
            # Solve with extended time limit
            solver = pl.PULP_CBC_CMD(timeLimit=time_limit, msg=1, gapRel=0.01)
            model.solve(solver)
            
            # Check if a feasible solution was found
            if model.status == pl.LpStatusOptimal or model.status == pl.LpStatusNotSolved:
                # Extract the solution
                schedule = []
                for s in range(1, num_staff+1):
                    for d in range(1, num_days+1):
                        for shift in possible_shifts:
                            if pl.value(x[(s, d, shift['id'])]) == 1:
                                # Find the shift details
                                shift_details = next((sh for sh in possible_shifts if sh['id'] == shift['id']), None)
                                
                                schedule.append({
                                    'staff_id': s,
                                    'day': d,
                                    'shift_id': shift['id'],
                                    'start': shift_details['start'],
                                    'end': shift_details['end'],
                                    'duration': shift_details['duration'],
                                    'cycles_covered': list(shift_details['cycles_covered'])
                                })
                
                return schedule, model.objective.value()
            else:
                return None, None
        except Exception as e:
            print(f"Error in optimization: {e}")
            return None, None
    
    # Try to solve with estimated number of staff
    if exact_staff_count is not None and exact_staff_count > 0:
        # If exact staff count is specified, only try with that count
        staff_count = int(exact_staff_count)
        results = f"Using exactly {staff_count} staff as specified...\n"
        
        # Try to solve with exactly this many staff
        schedule, objective = optimize_schedule(staff_count)
        
        if schedule is None:
            results += f"Failed to find a feasible solution with exactly {staff_count} staff.\n"
            results += "Try increasing the staff count.\n"
            return results, None, None, None, None
    else:
        # Start from theoretical minimum and work up
        min_staff = max(1, int(theoretical_min_staff))  # Start from theoretical minimum
        max_staff = int(min_staff_estimate) + 5  # Allow some buffer
        
        results = f"Theoretical minimum staff needed: {theoretical_min_staff:.1f}\n"
        results += f"Searching for minimum staff count starting from {min_staff}...\n"
        
        # Try each staff count from min to max
        for staff_count in range(min_staff, max_staff + 1):
            results += f"Trying with {staff_count} staff...\n"
            
            # Increase time limit for each attempt to give the solver more time
            time_limit = 300 + (staff_count - min_staff) * 100  # More time for larger staff counts
            schedule, objective = optimize_schedule(staff_count, time_limit)
            
            if schedule is not None:
                results += f"Found feasible solution with {staff_count} staff.\n"
                break
        
        if schedule is None:
            results += "Failed to find a feasible solution with the attempted staff counts.\n"
            results += "Try increasing the staff count manually or relaxing constraints.\n"
            return results, None, None, None, None
    
    results += f"Optimal solution found with {staff_count} staff\n"
    results += f"Total staff hours: {objective}\n"
    
    # Convert to DataFrame for analysis
    schedule_df = pd.DataFrame(schedule)
    
    # Analyze staff workload
    staff_hours = {}
    for s in range(1, staff_count+1):
        staff_shifts = schedule_df[schedule_df['staff_id'] == s]
        total_hours = staff_shifts['duration'].sum()
        staff_hours[s] = total_hours
    
    # After calculating staff hours, filter out staff with 0 hours before displaying
    active_staff_hours = {s: hours for s, hours in staff_hours.items() if hours > 0}
    
    results += "\nStaff Hours:\n"
    for staff_id, hours in active_staff_hours.items():
        utilization = (hours / MAX_HOURS_PER_STAFF) * 100
        results += f"Staff {staff_id}: {hours} hours ({utilization:.1f}% utilization)\n"
        # Add overtime information
        if hours > MAX_HOURS_PER_STAFF:
            overtime = hours - MAX_HOURS_PER_STAFF
            overtime_percent = (overtime / MAX_HOURS_PER_STAFF) * 100
            results += f"  Overtime: {overtime:.1f} hours ({overtime_percent:.1f}%)\n"
    
    # Use active_staff_hours for average utilization calculation
    active_staff_count = len(active_staff_hours)
    avg_utilization = sum(active_staff_hours.values()) / (active_staff_count * MAX_HOURS_PER_STAFF) * 100
    results += f"\nAverage staff utilization: {avg_utilization:.1f}%\n"
    
    # Check coverage for each day and cycle
    coverage_check = []
    for d in range(1, num_days+1):
        day_index = d - 1  # 0-indexed for DataFrame
        
        day_schedule = schedule_df[schedule_df['day'] == d]
        
        for cycle in cycle_cols:
            required = data.iloc[day_index][f'{cycle}_staff']
            
            # Count staff covering this cycle
            assigned = sum(1 for _, shift in day_schedule.iterrows() 
                          if cycle in shift['cycles_covered'])
            
            coverage_check.append({
                'day': d,
                'cycle': cycle,
                'required': required,
                'assigned': assigned,
                'satisfied': assigned >= required
            })
    
    coverage_df = pd.DataFrame(coverage_check)
    satisfaction = coverage_df['satisfied'].mean() * 100
    results += f"Coverage satisfaction: {satisfaction:.1f}%\n"
    
    if satisfaction < 100:
        results += "Warning: Not all staffing requirements are met!\n"
        unsatisfied = coverage_df[~coverage_df['satisfied']]
        results += unsatisfied.to_string() + "\n"
    
    # Generate detailed schedule report
    detailed_schedule = "Detailed Schedule:\n"
    for d in range(1, num_days+1):
        day_schedule = schedule_df[schedule_df['day'] == d]
        day_schedule = day_schedule.sort_values(['start'])
        
        detailed_schedule += f"\nDay {d}:\n"
        for _, shift in day_schedule.iterrows():
            start_hour = shift['start']
            end_hour = shift['end']

            start_str = am_pm(start_hour)
            end_str = am_pm(end_hour)

            cycles = ", ".join(shift['cycles_covered'])
            detailed_schedule += f"  Staff {shift['staff_id']}: {start_str}-{end_str} ({shift['duration']} hrs), Cycles: {cycles}\n"
    
    # Generate schedule visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Prepare schedule for plotting
    staff_days = {}
    for s in range(1, staff_count+1):
        staff_days[s] = [0] * num_days  # 0 means off duty
    
    for _, shift in schedule_df.iterrows():
        staff_id = shift['staff_id']
        day = shift['day'] - 1  # 0-indexed
        staff_days[staff_id][day] = shift['duration']
    
    # Plot the schedule
    for s, hours in staff_days.items():
        ax.bar(range(1, num_days+1), hours, label=f'Staff {s}')
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Shift Hours')
    ax.set_title('Staff Schedule')
    ax.set_xticks(range(1, num_days+1))
    ax.legend()
    
    # Save the figure to a temporary file
    plot_path = None
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.savefig(f.name)
        plt.close(fig)
        plot_path = f.name
    
    # Create a Gantt chart with advanced visuals and alternating labels - only showing active staff
    gantt_path = create_gantt_chart(schedule_df, num_days, staff_count)

    # Convert schedule to CSV data
    schedule_df['start_ampm'] = schedule_df['start'].apply(am_pm)
    schedule_df['end_ampm'] = schedule_df['end'].apply(am_pm)
    schedule_csv = schedule_df[['staff_id', 'day', 'start_ampm', 'end_ampm', 'duration', 'cycles_covered']].to_csv(index=False)

    # Create a temporary file and write the CSV data into it
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as temp_file:
        temp_file.write(schedule_csv)
        schedule_csv_path = temp_file.name

    # Create staff assignment table
    staff_assignment_data = []
    for d in range(1, num_days + 1):
        cycle_staff = {}
        for cycle in cycle_cols:
            # Get staff IDs assigned to this cycle on this day
            staff_ids = schedule_df[(schedule_df['day'] == d) & (schedule_df['cycles_covered'].apply(lambda x: cycle in x))]['staff_id'].tolist()
            cycle_staff[cycle] = len(staff_ids)
        staff_assignment_data.append([d] + [cycle_staff[cycle] for cycle in cycle_cols])

    staff_assignment_df = pd.DataFrame(staff_assignment_data, columns=['Day'] + cycle_cols)
    
    # Create CSV files for download
    staff_assignment_csv_path = None
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as temp_file:
        staff_assignment_df.to_csv(temp_file.name, index=False)
        staff_assignment_csv_path = temp_file.name
    
    # Return all required values in the correct order
    return results, staff_assignment_df, gantt_path, schedule_df, plot_path, schedule_csv_path, staff_assignment_csv_path

def convert_to_24h(time_str):
    """Converts AM/PM time string to 24-hour format."""
    try:
        time_obj = datetime.strptime(time_str, "%I:00 %p")
        return time_obj.hour
    except ValueError:
        return None

def gradio_wrapper(
    csv_file, beds_per_staff, max_hours_per_staff, hours_per_cycle,
    rest_days_per_week, clinic_start_ampm, clinic_end_ampm, overlap_time, max_start_time_change,
    exact_staff_count=None, overtime_percent=100
):
    try:
        # Convert AM/PM times to 24-hour format
        clinic_start = convert_to_24h(clinic_start_ampm)
        clinic_end = convert_to_24h(clinic_end_ampm)
        
        # Call the optimization function
        results, staff_assignment_df, gantt_path, schedule_df, plot_path, schedule_csv_path, staff_assignment_csv_path = optimize_staffing(
            csv_file, beds_per_staff, max_hours_per_staff, hours_per_cycle,
            rest_days_per_week, clinic_start, clinic_end, overlap_time, max_start_time_change,
            exact_staff_count, overtime_percent
        )
        
        # Return the results
        return staff_assignment_df, gantt_path, schedule_df, plot_path, staff_assignment_csv_path, schedule_csv_path
    except Exception as e:
        # If there's an error in the optimization process, return a meaningful error message
        empty_staff_df = pd.DataFrame(columns=["Day"])
        error_message = f"Error during optimization: {str(e)}\n\nPlease try with different parameters or a simpler dataset."
        # Return error in the first output
        return empty_staff_df, None, None, None, None, None

# Create a Gantt chart with advanced visuals and alternating labels - only showing active staff
def create_gantt_chart(schedule_df, num_days, staff_count):
    # Get the list of active staff IDs (staff who have at least one shift)
    active_staff_ids = sorted(schedule_df['staff_id'].unique())
    active_staff_count = len(active_staff_ids)
    
    # Create a mapping from original staff ID to position in the chart
    staff_position = {staff_id: i+1 for i, staff_id in enumerate(active_staff_ids)}
    
    # Create a larger figure with higher DPI
    plt.figure(figsize=(max(30, num_days * 1.5), max(12, active_staff_count * 0.8)), dpi=200)
    
    # Use a more sophisticated color palette - only for active staff
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, active_staff_count))
    
    # Set a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a new axis with a slight background color
    
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    # Sort by staff then day
    schedule_df = schedule_df.sort_values(['staff_id', 'day'])
    
    # Plot Gantt chart - only for active staff
    for i, staff_id in enumerate(active_staff_ids):
        staff_shifts = schedule_df[schedule_df['staff_id'] == staff_id]
        
        y_pos = active_staff_count - i  # Position based on index in active staff list
        
        # Add staff label with a background box
        ax.text(-0.7, y_pos, f"Staff {staff_id}", fontsize=12, fontweight='bold', 
                ha='right', va='center', bbox=dict(facecolor='white', edgecolor='gray', 
                                                  boxstyle='round,pad=0.5', alpha=0.9))
        
        # Add a subtle background for each staff row
        ax.axhspan(y_pos-0.4, y_pos+0.4, color='white', alpha=0.4, zorder=-5)
        
        # Track shift positions to avoid label overlap
        shift_positions = []
        
        for idx, shift in enumerate(staff_shifts.iterrows()):
            _, shift = shift
            day = shift['day']
            start_hour = shift['start']
            end_hour = shift['end']
            duration = shift['duration']
            
            # Format times for display
            start_ampm = am_pm(start_hour)
            end_ampm = am_pm(end_hour)
            
            # Calculate shift position
            shift_start_pos = day-1+start_hour/24
            
            # Handle overnight shifts
            if end_hour < start_hour:  # Overnight shift
                # First part of shift (until midnight)
                rect1 = ax.barh(y_pos, (24-start_hour)/24, left=shift_start_pos, 
                       height=0.6, color=colors[i], alpha=0.9, 
                       edgecolor='black', linewidth=1, zorder=10)
                
                # Add gradient effect
                for r in rect1:
                    r.set_edgecolor('black')
                    r.set_linewidth(1)
                
                # Second part of shift (after midnight)
                rect2 = ax.barh(y_pos, end_hour/24, left=day, 
                       height=0.6, color=colors[i], alpha=0.9,
                       edgecolor='black', linewidth=1, zorder=10)
                
                # Add gradient effect
                for r in rect2:
                    r.set_edgecolor('black')
                    r.set_linewidth(1)
                
                # For overnight shifts, we'll place the label in the first part if it's long enough
                shift_width = (24-start_hour)/24
                if shift_width >= 0.1:  # Only add label if there's enough space
                    label_pos = shift_start_pos + shift_width/2
                    
                    # Alternate labels above and below
                    y_offset = 0.35 if idx % 2 == 0 else -0.35
                    
                    # Add label with background for better readability
                    label = f"{start_ampm}-{end_ampm}"
                    text = ax.text(label_pos, y_pos + y_offset, label, 
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color='black', bbox=dict(facecolor='white', alpha=0.9, pad=3, 
                                                   boxstyle='round,pad=0.3', edgecolor='gray'),
                           zorder=20)
                    
                    shift_positions.append(label_pos)
            else:
                # Regular shift
                shift_width = duration/24
                rect = ax.barh(y_pos, shift_width, left=shift_start_pos, 
                       height=0.6, color=colors[i], alpha=0.9,
                       edgecolor='black', linewidth=1, zorder=10)
                
                # Add gradient effect
                for r in rect:
                    r.set_edgecolor('black')
                    r.set_linewidth(1)
                
                # Only add label if there's enough space
                if shift_width >= 0.1:
                    label_pos = shift_start_pos + shift_width/2
                    
                    # Alternate labels above and below
                    y_offset = 0.35 if idx % 2 == 0 else -0.35
                    
                    # Add label with background for better readability
                    label = f"{start_ampm}-{end_ampm}"
                    text = ax.text(label_pos, y_pos + y_offset, label, 
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color='black', bbox=dict(facecolor='white', alpha=0.9, pad=3, 
                                                   boxstyle='round,pad=0.3', edgecolor='gray'),
                           zorder=20)
                    
                    shift_positions.append(label_pos)
    
    # Add weekend highlighting with a more sophisticated look
    for day in range(1, num_days + 1):
        # Determine if this is a weekend (assuming day 1 is Monday)
        is_weekend = (day % 7 == 0) or (day % 7 == 6)  # Saturday or Sunday
        
        if is_weekend:
            ax.axvspan(day-1, day, alpha=0.15, color='#ff9999', zorder=-10)
            day_label = "Saturday" if day % 7 == 6 else "Sunday"
            ax.text(day-0.5, 0.2, day_label, ha='center', fontsize=10, color='#cc0000',
                   fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=2, boxstyle='round'))
    
    # Set x-axis ticks for each day with better formatting
    ax.set_xticks(np.arange(0.5, num_days, 1))
    day_labels = [f"Day {d}" for d in range(1, num_days+1)]
    ax.set_xticklabels(day_labels, rotation=0, ha='center', fontsize=10)
    
    # Add vertical lines between days with better styling
    for day in range(1, num_days):
        ax.axvline(x=day, color='#aaaaaa', linestyle='-', alpha=0.5, zorder=-5)
    
    # Set y-axis ticks for each staff
    ax.set_yticks(np.arange(1, active_staff_count+1))
    ax.set_yticklabels([])  # Remove default labels as we've added custom ones
    
    # Set axis limits with some padding
    ax.set_xlim(-0.8, num_days)
    ax.set_ylim(0.5, active_staff_count + 0.5)
    
    # Add grid for hours (every 6 hours) with better styling
    for day in range(num_days):
        for hour in [6, 12, 18]:
            ax.axvline(x=day + hour/24, color='#cccccc', linestyle=':', alpha=0.5, zorder=-5)
            # Add small hour markers at the bottom
            hour_label = "6AM" if hour == 6 else "Noon" if hour == 12 else "6PM"
            ax.text(day + hour/24, 0, hour_label, ha='center', va='bottom', fontsize=7, 
                   color='#666666', rotation=90, alpha=0.7)
    
    # Add title and labels with more sophisticated styling
    plt.title(f'Staff Schedule ({active_staff_count} Active Staff)', fontsize=24, fontweight='bold', pad=20, color='#333333')
    plt.xlabel('Day', fontsize=16, labelpad=10, color='#333333')
    
    # Add a legend for time reference with better styling
    time_box = plt.figtext(0.01, 0.01, "Time Reference:", ha='left', fontsize=10, 
                          fontweight='bold', color='#333333')
    time_markers = ['6 AM', 'Noon', '6 PM', 'Midnight']
    for i, time in enumerate(time_markers):
        plt.figtext(0.08 + i*0.06, 0.01, time, ha='left', fontsize=9, color='#555555')
    
    # Remove spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    
    # Add a note about weekends with better styling
    weekend_note = plt.figtext(0.01, 0.97, "Red areas = Weekends", fontsize=12, 
                              color='#cc0000', fontweight='bold',
                              bbox=dict(facecolor='white', alpha=0.7, pad=5, boxstyle='round'))
    
    # Add a subtle border around the entire chart
    plt.box(False)
    
    # Save the Gantt chart with high quality
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.tight_layout()
        plt.savefig(f.name, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return f.name

# Define Gradio UI
am_pm_times = [f"{i:02d}:00 AM" for i in range(1, 13)] + [f"{i:02d}:00 PM" for i in range(1, 13)]

# Add CSS for chart containers
css = """
.chart-container {
    height: 800px !important;
    width: 100% !important;
    margin: 20px 0;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.weekly-chart-container {
    height: 1000px !important;
    width: 100% !important;
    margin: 20px 0;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Ensure plotly charts are visible */
.js-plotly-plot {
    width: 100% !important;
    height: 100% !important;
}

/* Improve visibility of chart titles */
.gtitle {
    font-weight: bold !important;
    font-size: 20px !important;
}
"""

with gr.Blocks(title="Staff Scheduling Optimizer", css=css) as iface:
    
    gr.Markdown("# Staff Scheduling Optimizer")
    gr.Markdown("Upload a CSV file with cycle data and configure parameters to generate an optimal staff schedule.")
    
    with gr.Row():
        # LEFT PANEL - Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Input Parameters")
            
            # Input parameters
            csv_input = gr.File(label="Upload CSV")
            beds_per_staff = gr.Number(label="Beds per Staff", value=3)
            max_hours_per_staff = gr.Number(label="Maximum monthly hours", value=160)
            hours_per_cycle = gr.Number(label="Hours per Cycle", value=4)
            rest_days_per_week = gr.Number(label="Rest Days per Week", value=2)
            clinic_start_ampm = gr.Dropdown(label="Clinic Start Hour (AM/PM)", choices=am_pm_times, value="08:00 AM")
            clinic_end_ampm = gr.Dropdown(label="Clinic End Hour (AM/PM)", choices=am_pm_times, value="08:00 PM")
            overlap_time = gr.Number(label="Overlap Time", value=0)
            max_start_time_change = gr.Number(label="Max Start Time Change", value=2)
            exact_staff_count = gr.Number(label="Exact Staff Count (optional)", value=None)
            overtime_percent = gr.Slider(label="Overtime Allowed (%)", minimum=0, maximum=100, value=100, step=10)
            
            optimize_btn = gr.Button("Optimize Schedule", variant="primary", size="lg")
        
        # RIGHT PANEL - Outputs
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            
            # Tabs for different outputs - reordered
            with gr.Tabs():
                with gr.TabItem("Detailed Schedule"):
                    with gr.Row():
                        csv_schedule = gr.Dataframe(label="Detailed Schedule", elem_id="csv_schedule")
                    
                    with gr.Row():
                        schedule_download_file = gr.File(label="Download Detailed Schedule", visible=True)
                
                with gr.TabItem("Gantt Chart"):
                    gantt_chart = gr.Image(label="Staff Schedule Visualization", elem_id="gantt_chart")
                
                with gr.TabItem("Staff Coverage by Cycle"):
                    with gr.Row():
                        staff_assignment_table = gr.Dataframe(label="Staff Count in Each Cycle (Staff May Overlap)", elem_id="staff_assignment_table")
                    
                    with gr.Row():
                        staff_download_file = gr.File(label="Download Coverage Table", visible=True)
                
                with gr.TabItem("Constraints and Analytics"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Applied Constraints")
                            constraints_text = gr.TextArea(
                                label="", 
                                interactive=False,
                                show_label=False
                            )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Monthly Distribution")
                                monthly_chart = gr.HTML(
                                    label="Monthly Hours Distribution",
                                    show_label=False,
                                    elem_classes="chart-container"
                                )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Weekly Distribution")
                                weekly_charts = gr.HTML(
                                    label="Weekly Hours Distribution",
                                    show_label=False,
                                    elem_classes="weekly-chart-container"
                                )
                
                with gr.TabItem("Staff Overlap"):
                    with gr.Row():
                        overlap_chart = gr.HTML(
                            label="Staff Overlap Visualization",
                            show_label=False
                        )
                    with gr.Row():
                        gr.Markdown("""
                        This heatmap shows the number of staff members working simultaneously throughout each day.
                        - Darker colors indicate more staff overlap
                        - The x-axis shows time of day in 30-minute intervals
                        - The y-axis shows each day of the schedule
                        """)
                
                with gr.TabItem("Staff Absence Handler"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Handle Staff Absence")
                            absent_staff = gr.Number(label="Staff ID to be absent", precision=0)
                            absence_start = gr.Number(label="Start Day", precision=0)
                            absence_end = gr.Number(label="End Day", precision=0)
                            handle_absence_btn = gr.Button("Redistribute Shifts", variant="primary")
                        
                        with gr.Column():
                            absence_result = gr.TextArea(label="Redistribution Results", interactive=False)
                            updated_schedule = gr.DataFrame(label="Updated Schedule")
                            absence_gantt_chart = gr.Image(label="Absence Schedule Visualization", elem_id="absence_gantt_chart")
    
    # Define download functions
    def create_download_link(df, filename="data.csv"):
        """Create a CSV download link for a dataframe"""
        if df is None or df.empty:
            return None
        
        csv_data = df.to_csv(index=False)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_data)
            return f.name

    # Update the optimize_and_display function
    def optimize_and_display(csv_file, beds_per_staff, max_hours_per_staff, hours_per_cycle,
                            rest_days_per_week, clinic_start_ampm, clinic_end_ampm, 
                            overlap_time, max_start_time_change, exact_staff_count, overtime_percent):
        try:
            # Convert AM/PM times to 24-hour format
            clinic_start = convert_to_24h(clinic_start_ampm)
            clinic_end = convert_to_24h(clinic_end_ampm)
            
            # Call the optimization function
            results, staff_assignment_df, gantt_path, schedule_df, plot_path, schedule_csv_path, staff_assignment_csv_path = optimize_staffing(
                csv_file, beds_per_staff, max_hours_per_staff, hours_per_cycle,
                rest_days_per_week, clinic_start, clinic_end, overlap_time, max_start_time_change,
                exact_staff_count, overtime_percent
            )
            
            if schedule_df is not None:
                try:
                    # Generate analytics data
                    constraints_info = get_constraints_summary(
                        max_hours_per_staff, 
                        rest_days_per_week,
                        overtime_percent
                    )
                    
                    # Create visualizations directly as HTML
                    monthly_html = create_monthly_distribution_chart(schedule_df)
                    weekly_html = create_weekly_distribution_charts(schedule_df)
                    overlap_html = create_overlap_visualization(schedule_df)
                    
                    return (
                        staff_assignment_df,
                        gantt_path,
                        schedule_df,
                        schedule_csv_path,
                        constraints_info,
                        monthly_html,
                        weekly_html,
                        overlap_html
                    )
                except Exception as e:
                    print(f"Error in visualization: {str(e)}")
                    return (
                        staff_assignment_df,
                        gantt_path,
                        schedule_df,
                        schedule_csv_path,
                        "Error in constraints",
                        "<div>Error creating monthly chart</div>",
                        "<div>Error creating weekly charts</div>",
                        "<div>Error creating overlap visualization</div>"
                    )
            else:
                return (None,) * 8
            
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            return (None,) * 8
    
    def get_constraints_summary(max_hours, rest_days, overtime_percent):
        """Generate a summary of all applied constraints from actual parameters"""
        constraints = [
            "Applied Scheduling Constraints:",
            "----------------------------",
            f"1. Maximum Hours per Month: {max_hours} hours",
            f"2. Required Rest Days per Week: {rest_days} days",
            f"3. Maximum Weekly Hours: 60 hours per staff member",
            "4. Minimum Rest Period: 11 hours between shifts",
            "5. Maximum Consecutive Days: 6 working days",
            f"6. Overtime Allowance: {overtime_percent}% of standard hours",
            "7. Coverage Requirements:",
            "   - All cycles must be fully staffed",
            "   - No understaffing allowed",
            "   - Staff assigned based on required beds/staff ratio",
            "8. Shift Constraints:",
            "   - Available shift durations: 5, 10 hours",
            "   - Shifts must align with cycle times",
            "9. Staff Scheduling Rules:",
            "   - Equal distribution of workload when possible",
            "   - Consistent shift patterns preferred",
            "   - Weekend rotations distributed fairly"
        ]
        return "\n".join(constraints)
    
    def create_monthly_distribution_chart(schedule_df):
        """Create Seaborn pie chart for monthly hours distribution"""
        if schedule_df is None or schedule_df.empty:
            return "<div>No data available for visualization</div>"
        
        try:
            # Calculate total hours per staff member
            staff_hours = schedule_df.groupby('staff_id')['duration'].sum()
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.set_palette("pastel")
            ax.pie(staff_hours, labels=staff_hours.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title("Monthly Hours Distribution")
            
            # Convert plot to PNG image
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')  # Added bbox_inches='tight'
            plt.close(fig)
            img.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(img.read()).decode('utf-8')
            img_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; max-height:600px;">'
            
            return img_html
        except Exception as e:
            print(f"Error in monthly chart: {e}")
            return f"<div>Error creating monthly chart: {str(e)}</div>"
    
    def create_weekly_distribution_charts(schedule_df):
        """Create Plotly pie charts for weekly hours distribution"""
        if schedule_df is None or schedule_df.empty:
            return "<div>No data available for visualization</div>"
        
        try:
            # Calculate total hours per staff member for each week
            schedule_df['week'] = schedule_df['day'] // 7  # Assuming each week starts on day 0, 7, 14, etc.
            weekly_hours = schedule_df.groupby(['week', 'staff_id'])['duration'].sum().reset_index()
            
            # Create staff labels
            weekly_hours['staff_label'] = weekly_hours.apply(
                lambda x: f"Staff {x['staff_id']} ({x['duration']:.1f}hrs)",
                axis=1
            )
            
            # Get unique weeks
            weeks = sorted(weekly_hours['week'].unique())
            
            # Define color palette
            colors = px.colors.qualitative.Set3
            
            # Create subplots
            fig = make_subplots(
                rows=len(weeks),
                cols=1,
                subplot_titles=[f'Week {week}' for week in weeks],
                specs=[[{'type': 'domain'}] for week in weeks]
            )
            
            # Add pie charts for each week
            for i, week in enumerate(weeks, start=1):
                week_data = weekly_hours[weekly_hours['week'] == week]
                
                fig.add_trace(
                    go.Pie(
                        values=week_data['duration'],
                        labels=week_data['staff_label'],
                        name=f'Week {week}',
                        showlegend=(i == 1),
                        marker_colors=colors,
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate=(
                            "Staff: %{label}<br>"
                            "Hours: %{value:.1f}<br>"
                            "Percentage: %{percent:.1f}%"
                            "<extra></extra>"
                        )
                    ),
                    row=i,
                    col=1
                )
            
            fig.update_layout(
                height=300 * len(weeks),
                width=800,
                title_text="Weekly Hours Distribution",
                title_x=0.5,
                title_font_size=20,
                margin=dict(t=50, l=50, r=50, b=50),
                showlegend=True
            )
            
            return fig.to_html(include_plotlyjs='cdn', full_html=False)
        except Exception as e:
            print(f"Error in weekly charts: {e}")
            return f"<div>Error creating weekly charts: {str(e)}</div>"

    # Add this new function for creating the overlap visualization
    def create_overlap_visualization(schedule_df):
        """Create Seaborn heatmap for staff overlap"""
        if schedule_df is None or schedule_df.empty:
            return "<div>No data available for visualization</div>"
        
        try:
            # Create 24-hour timeline with 30-minute intervals
            intervals = 48  # 24 hours * 2 (30-minute intervals)
            days = sorted(schedule_df['day'].unique())
            
            # Initialize overlap matrix
            overlap_data = np.zeros((len(days), intervals))
            
            # Calculate overlaps
            for day_idx, day in enumerate(days):
                day_shifts = schedule_df[schedule_df['day'] == day]
                
                for i in range(intervals):
                    time = i * 0.5
                    staff_working = 0
                    
                    for _, shift in day_shifts.iterrows():
                        start = shift['start']
                        end = shift['end']
                        
                        if end < start:  # Overnight shift
                            if time >= start or time < end:
                                staff_working += 1
                        else:
                            if start <= time < end:
                                staff_working += 1
                
                overlap_data[day_idx, i] = staff_working
            
            # Create time labels
            time_labels = [f"{int(i//2):02d}:{int((i%2)*30):02d}" for i in range(intervals)]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(overlap_data, cmap="viridis", ax=ax, cbar_kws={'label': 'Staff Count'})
            
            # Set labels
            ax.set_xticks(np.arange(len(time_labels[::4])))
            ax.set_xticklabels(time_labels[::4], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(days)))
            ax.set_yticklabels(days)
            
            # Add title
            ax.set_title("Staff Overlap Throughout the Day")
            
            # Ensure layout is tight
            plt.tight_layout()
            
            # Convert plot to PNG image
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')  # Added bbox_inches='tight'
            plt.close(fig)
            img.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(img.read()).decode('utf-8')
            img_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; max-height:800px;">'
            
            return img_html
        except Exception as e:
            print(f"Error in overlap visualization: {e}")
            return f"<div>Error creating overlap visualization: {str(e)}</div>"

    # Connect the button to the optimization function
    optimize_btn.click(
        fn=optimize_and_display,
        inputs=[
            csv_input, beds_per_staff, max_hours_per_staff, hours_per_cycle,
            rest_days_per_week, clinic_start_ampm, clinic_end_ampm, 
            overlap_time, max_start_time_change, exact_staff_count, overtime_percent
        ],
        outputs=[
            staff_assignment_table,  # Staff coverage table
            gantt_chart,            # Gantt chart
            csv_schedule,           # Detailed schedule
            schedule_download_file,   # Download file
            constraints_text,        # Constraints text
            monthly_chart,          # Monthly distribution
            weekly_charts,          # Weekly distribution
            overlap_chart           # Staff overlap visualization
        ]
    )

    # Add the handler function
    def handle_absence_click(staff_id, start_day, end_day, current_schedule, max_hours_per_staff, overtime_percent):
        if current_schedule is None or current_schedule.empty:
            return "No current schedule loaded.", None, None
        
        absence_dates = list(range(int(start_day), int(end_day) + 1))
        summary, absence_schedule, absence_gantt_path = handle_staff_absence(
            current_schedule,
            int(staff_id),
            absence_dates,
            max_hours_per_staff,
            overtime_percent
        )
        
        return summary, absence_schedule, absence_gantt_path

    # Connect the absence handler button
    handle_absence_btn.click(
        fn=handle_absence_click,
        inputs=[
            absent_staff,
            absence_start,
            absence_end,
            csv_schedule,  # Current schedule
            max_hours_per_staff,  # Add this parameter
            overtime_percent  # Add this parameter
        ],
        outputs=[
            absence_result,
            updated_schedule,
            absence_gantt_chart
        ]
    )

# Launch the Gradio app
iface.launch(share=True)

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# NEF Scheduling System")
        
        with gr.Tabs() as tabs:
            with gr.Tab("Schedule Input"):
                # Schedule input components
                with gr.Row():
                    csv_input = gr.File(label="Upload Schedule Data (CSV)")
                    schedule_preview = gr.DataFrame(label="Schedule Preview")
            
            with gr.Tab("Schedule Output"):
                # Schedule output components
                with gr.Row():
                    schedule_output = gr.DataFrame(label="Generated Schedule")
                    download_btn = gr.Button("Download Schedule")
            
            with gr.Tab("Constraints and Analytics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Applied Constraints")
                        constraints_text = gr.TextArea(label="", interactive=False)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Monthly Distribution")
                            monthly_chart = gr.HTML(label="Monthly Hours Distribution")
                            
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Weekly Distribution")
                            weekly_charts = gr.HTML(label="Weekly Hours Distribution")

        with gr.TabItem("Staff Absence Handler"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Handle Staff Absence")
                    absent_staff = gr.Number(label="Staff ID to be absent", precision=0)
                    absence_start = gr.Number(label="Start Day", precision=0)
                    absence_end = gr.Number(label="End Day", precision=0)
                    handle_absence_btn = gr.Button("Redistribute Shifts", variant="primary")
                
                with gr.Column():
                    absence_result = gr.TextArea(label="Redistribution Results", interactive=False)
                    updated_schedule = gr.DataFrame(label="Updated Schedule")
                    absence_gantt_chart = gr.Image(label="Absence Schedule Visualization", elem_id="absence_gantt_chart")

    return demo

def handle_staff_absence(schedule_df, absent_staff_id, absence_dates, max_hours_per_staff, overtime_percent):
    """
    Redistribute shifts of absent staff member to others, prioritizing staff with lowest monthly hours
    """
    try:
        # Create a copy of the original schedule
        new_schedule = schedule_df.copy()
        
        # Get shifts that need to be redistributed
        absent_shifts = new_schedule[
            (new_schedule['staff_id'] == absent_staff_id) & 
            (new_schedule['day'].isin(absence_dates))
        ]
        
        if absent_shifts.empty:
            return "No shifts found for the specified staff member on given dates.", None, None
        
        # Get available staff (excluding absent staff)
        available_staff = sorted(list(set(new_schedule['staff_id']) - {absent_staff_id}))
        
        # Calculate current hours for each staff member
        current_hours = new_schedule.groupby('staff_id')['duration'].sum()
        
        # Sort staff by current hours (ascending) to prioritize those with fewer hours
        staff_hours_sorted = current_hours.reindex(available_staff).sort_values()
        available_staff = staff_hours_sorted.index.tolist()
        
        # Calculate remaining available hours for each staff
        max_allowed_hours = max_hours_per_staff * (1 + overtime_percent/100)
        available_hours = {
            staff_id: max_allowed_hours - current_hours.get(staff_id, 0)
            for staff_id in available_staff
        }
        
        results = []
        unassigned_shifts = []
        
        # Process each shift that needs to be redistributed
        for _, shift in absent_shifts.iterrows():
            # Find eligible staff for this shift, prioritizing those with fewer hours
            eligible_staff = []
            eligible_staff_hours = {}
            
            for staff_id in available_staff:
                # Check if staff has enough remaining hours
                if available_hours[staff_id] >= shift['duration']:
                    # Check if staff is not already working that day
                    staff_shifts_that_day = new_schedule[
                        (new_schedule['staff_id'] == staff_id) & 
                        (new_schedule['day'] == shift['day'])
                    ]
                    
                    if staff_shifts_that_day.empty:
                        # Check minimum rest period (11 hours)
                        day_before = new_schedule[
                            (new_schedule['staff_id'] == staff_id) & 
                            (new_schedule['day'] == shift['day'] - 1)
                        ]
                        
                        day_after = new_schedule[
                            (new_schedule['staff_id'] == staff_id) & 
                            (new_schedule['day'] == shift['day'] + 1)
                        ]
                        
                        can_work = True
                        if not day_before.empty:
                            end_time_before = day_before.iloc[0]['end']
                            if (shift['start'] + 24 - end_time_before) < 11:
                                can_work = False
                        
                        if not day_after.empty and can_work:
                            start_time_after = day_after.iloc[0]['start']
                            if (starttime_after + 24 - shift['end']) < 11:
                                can_work = False
                        
                        if can_work:
                            eligible_staff.append(staff_id)
                            eligible_staff_hours[staff_id] = current_hours.get(staff_id, 0)
            
            if eligible_staff:
                # Sort eligible staff by current hours to prioritize those with fewer hours
                sorted_eligible = sorted(eligible_staff, key=lambda x: eligible_staff_hours[x])
                best_staff = sorted_eligible[0]  # Select staff with lowest hours
                
                # Update the schedule
                new_schedule.loc[shift.name, 'staff_id'] = best_staff
                
                # Update available hours and current hours
                available_hours[best_staff] -= shift['duration']
                current_hours[best_staff] = current_hours.get(best_staff, 0) + shift['duration']
                
                results.append(
                    f"Shift on Day {shift['day']} ({shift['duration']} hours) "
                    f"reassigned to Staff {best_staff} (current hours: {current_hours[best_staff]:.1f})"
                )
            else:
                unassigned_shifts.append(
                    f"Could not reassign shift on Day {shift['day']} ({shift['duration']} hours)"
                )
        
        # Generate detailed summary with hours distribution
        summary = "\n".join([
            "Shift Redistribution Summary:",
            "----------------------------",
            f"Staff {absent_staff_id} absent for {len(absence_dates)} days",
            f"Successfully reassigned: {len(results)} shifts",
            f"Failed to reassign: {len(unassigned_shifts)} shifts",
            "\nCurrent Hours Distribution:",
            "-------------------------"
        ] + [
            f"Staff {s}: {current_hours.get(s, 0):.1f} hours (of max {max_allowed_hours:.1f})"
            for s in sorted(available_staff)
        ] + [
            "\nReassignment Details:",
            *results,
            "\nUnassigned Shifts:",
            *unassigned_shifts
        ])
        
        # Filter the schedule for the absence period
        absence_schedule = new_schedule[new_schedule['day'].isin(absence_dates)].copy()
        
        # Create a Gantt chart for the absence period
        absence_gantt_path = create_gantt_chart(absence_schedule, len(absence_dates), len(set(absence_schedule['staff_id'])))
        
        if unassigned_shifts:
            return summary, None, None
        else:
            return summary, absence_schedule, absence_gantt_path
            
    except Exception as e:
        return f"Error redistributing shifts: {str(e)}", None, None

class FastScheduler:
    def __init__(self, num_staff, num_days, possible_shifts, staff_requirements, constraints):
        self.num_staff = num_staff
        self.num_days = num_days
        self.possible_shifts = possible_shifts
        self.staff_requirements = staff_requirements
        self.constraints = constraints
        self.best_schedule = None
        self.best_score = float('inf')
        self.best_staff_count = float('inf')
        self.cache = {}  # Cache for memoization

    def optimize(self, time_limit=60):
        """Enhanced optimization with staff minimization"""
        start_time = time.time()
        current_staff = self.num_staff
        
        while time.time() - start_time < time_limit and current_staff > 1:
            # Try to find solution with current staff count
            schedule = self._construct_initial_schedule(current_staff)
            if not schedule:
                # If no feasible schedule found, we've reached minimum staff
                break
                
            improved_schedule = self._parallel_improvement(schedule, time_limit - (time.time() - start_time))
            score = self._evaluate_schedule(improved_schedule)
            staff_used = len(set(shift['staff_id'] for shift in improved_schedule))
            
            if score < float('inf'):  # Feasible solution found
                if staff_used < self.best_staff_count or (staff_used == self.best_staff_count and score < self.best_score):
                    self.best_schedule = improved_schedule.copy()
                    self.best_score = score
                    self.best_staff_count = staff_used
                    # Try reducing staff count further
                    current_staff = staff_used - 1
                else:
                    break  # No improvement possible
            else:
                break  # No feasible solution with current staff count
        
        return self.best_schedule

    def _construct_initial_schedule(self, max_staff):
        """Improved initial schedule construction"""
        schedule = []
        staff_hours = {s: 0 for s in range(1, max_staff + 1)}
        staff_last_shift = {s: None for s in range(1, max_staff + 1)}
        
        # Pre-calculate daily requirements
        day_requirements = []
        for d in range(1, self.num_days + 1):
            total_required = sum(self.staff_requirements[d-1].values())
            cycles_required = self.staff_requirements[d-1]
            day_requirements.append((d, total_required, cycles_required))
        
        # Sort days by complexity (required staff and cycle distribution)
        day_requirements.sort(key=lambda x: (x[1], -len(x[2])), reverse=True)
        
        # Process each day
        for day, _, cycles in day_requirements:
            # Sort cycles by staff needed
            sorted_cycles = sorted(cycles.items(), key=lambda x: x[1], reverse=True)
            
            for cycle, staff_needed in sorted_cycles:
                staff_assigned = 0
                available_staff = self._get_available_staff(day, staff_hours, staff_last_shift, max_staff)
                
                for staff_id in available_staff:
                    if staff_assigned >= staff_needed:
                        break
                        
                    shift = self._find_optimal_shift(staff_id, day, cycle, staff_hours)
                    if shift:
                        schedule.append(shift)
                        staff_hours[staff_id] += shift['duration']
                        staff_last_shift[staff_id] = day
                        staff_assigned += 1
                
                if staff_assigned < staff_needed:
                    return []  # Infeasible with current staff count
        
        return schedule

    def _get_available_staff(self, day, staff_hours, staff_last_shift, max_staff):
        """Get available staff sorted by optimal criteria"""
        staff_scores = []
        week_start = ((day - 1) // 7) * 7 + 1
        
        for staff_id in range(1, max_staff + 1):
            if not self._can_assign_shift(staff_id, day, staff_hours, staff_last_shift):
                continue
                
            # Calculate score based on multiple factors
            hours_score = 1 / (staff_hours[staff_id] + 1)  # Prefer less utilized staff
            last_shift_gap = day - (staff_last_shift[staff_id] or 0)
            rest_score = min(last_shift_gap, 7) / 7  # Prefer well-rested staff
            
            # Calculate weekly hours
            week_hours = sum(shift['duration'] 
                           for shift in (self.best_schedule or [])
                           if shift['staff_id'] == staff_id 
                           and week_start <= shift['day'] < week_start + 7)
            week_score = 1 - (week_hours / 40)  # Prefer staff with fewer hours this week
            
            total_score = hours_score * 0.5 + rest_score * 0.3 + week_score * 0.2
            staff_scores.append((staff_id, total_score))
        
        return [s[0] for s in sorted(staff_scores, key=lambda x: x[1], reverse=True)]

    def _find_optimal_shift(self, staff_id, day, required_cycle, staff_hours):
        """Find the best shift considering multiple optimization criteria"""
        cache_key = (staff_id, day, required_cycle)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        valid_shifts = [
            shift for shift in self.possible_shifts
            if required_cycle in shift['cycles_covered']
        ]
        
        if not valid_shifts:
            return None
        
        # Get previous shift patterns
        prev_shifts = [
            shift for shift in (self.best_schedule or [])
            if shift['staff_id'] == staff_id 
            and shift['day'] < day 
            and shift['day'] >= day - 7
        ]
        
        best_shift = None
        best_score = float('inf')
        
        for shift in valid_shifts:
            score = 0
            
            # Prefer shifts that maintain consistent timing
            if prev_shifts:
                avg_start = sum(s['start'] for s in prev_shifts) / len(prev_shifts)
                score += abs(shift['start'] - avg_start) * 2
            
            # Prefer shorter shifts when staff has high hours
            hours_ratio = staff_hours[staff_id] / self.constraints['max_hours']
            score += shift['duration'] * hours_ratio
            
            # Prefer shifts that cover more required cycles
            coverage_score = -len(shift['cycles_covered'])
            score += coverage_score
            
            if score < best_score:
                best_score = score
                best_shift = shift
        
        if best_shift:
            result = {
                'staff_id': staff_id,
                'day': day,
                'shift_id': best_shift['id'],
                'start': best_shift['start'],
                'end': best_shift['end'],
                'duration': best_shift['duration'],
                'cycles_covered': list(best_shift['cycles_covered'])
            }
            self.cache[cache_key] = result
            return result
        
        return None

    def _parallel_improvement(self, schedule, remaining_time):
        """Parallel local search improvement"""
        improved = schedule.copy()
        improvement_found = True
        start_time = time.time()
        
        while improvement_found and time.time() - start_time < remaining_time:
            improvement_found = False
            
            # Try different improvement strategies
            strategies = [
                self._swap_shifts,
                self._reassign_shift,
                self._merge_shifts
            ]
            
            for strategy in strategies:
                if time.time() - start_time >= remaining_time:
                    break
                    
                new_schedule = strategy(improved)
                if new_schedule:
                    new_score = self._evaluate_schedule(new_schedule)
                    if new_score < self._evaluate_schedule(improved):
                        improved = new_schedule
                        improvement_found = True
                        break
        
        return improved

    def _swap_shifts(self, schedule):
        """Swap shifts between staff members"""
        best_schedule = None
        best_score = self._evaluate_schedule(schedule)
        
        for i in range(len(schedule)):
            for j in range(i + 1, len(schedule)):
                if schedule[i]['day'] == schedule[j]['day']:
                    new_schedule = schedule.copy()
                    new_schedule[i]['staff_id'], new_schedule[j]['staff_id'] = \
                        new_schedule[j]['staff_id'], new_schedule[i]['staff_id']
                    
                    score = self._evaluate_schedule(new_schedule)
                    if score < best_score:
                        best_schedule = new_schedule
                        best_score = score
        
        return best_schedule

    def _reassign_shift(self, schedule):
        """Try to reassign shifts to minimize staff"""
        best_schedule = None
        best_score = self._evaluate_schedule(schedule)
        
        staff_ids = sorted(set(s['staff_id'] for s in schedule))
        for shift in schedule:
            original_staff = shift['staff_id']
            for new_staff in staff_ids:
                if new_staff != original_staff:
                    new_schedule = schedule.copy()
                    shift_idx = schedule.index(shift)
                    new_schedule[shift_idx] = dict(shift, staff_id=new_staff)
                    
                    score = self._evaluate_schedule(new_schedule)
                    if score < best_score:
                        best_schedule = new_schedule
                        best_score = score
        
        return best_schedule

    def _merge_shifts(self, schedule):
        """Try to merge consecutive shifts for the same staff"""
        best_schedule = None
        best_score = self._evaluate_schedule(schedule)
        
        # Sort shifts by staff and day
        sorted_shifts = sorted(schedule, key=lambda x: (x['staff_id'], x['day'], x['start']))
        
        for i in range(len(sorted_shifts) - 1):
            shift1 = sorted_shifts[i]
            shift2 = sorted_shifts[i + 1]
            
            if (shift1['staff_id'] == shift2['staff_id'] and 
                shift1['day'] == shift2['day'] and 
                shift1['end'] == shift2['start']):
                
                # Try to merge shifts
                new_schedule = [s for s in schedule if s != shift1 and s != shift2]
                merged_shift = {
                    'staff_id': shift1['staff_id'],
                    'day': shift1['day'],
                    'shift_id': f"merged_{shift1['shift_id']}_{shift2['shift_id']}",
                    'start': shift1['start'],
                    'end': shift2['end'],
                    'duration': shift1['duration'] + shift2['duration'],
                    'cycles_covered': list(set(shift1['cycles_covered'] + shift2['cycles_covered']))
                }
                new_schedule.append(merged_shift)
                
                score = self._evaluate_schedule(new_schedule)
                if score < best_score:
                    best_schedule = new_schedule
                    best_score = score
        
        return best_schedule

    def _evaluate_schedule(self, schedule):
        """Fast schedule evaluation with caching"""
        # Convert schedule to hashable format for caching
        schedule_key = tuple(sorted((s['staff_id'], s['day'], s['shift_id']) for s in schedule))
        if schedule_key in self.cache:
            return self.cache[schedule_key]
        
        score = 0
        staff_hours = {}
        daily_coverage = {}
        
        # Process all shifts once to build necessary data structures
        for shift in schedule:
            staff_id = shift['staff_id']
            day = shift['day']
            
            # Update staff hours
            if staff_id not in staff_hours:
                staff_hours[staff_id] = {'total': 0, 'weekly': {}}
            staff_hours[staff_id]['total'] += shift['duration']
            
            week = (day - 1) // 7
            if week not in staff_hours[staff_id]['weekly']:
                staff_hours[staff_id]['weekly'][week] = 0
            staff_hours[staff_id]['weekly'][week] += shift['duration']
            
            # Update daily coverage
            if day not in daily_coverage:
                daily_coverage[day] = {}
            for cycle in shift['cycles_covered']:
                daily_coverage[day][cycle] = daily_coverage[day].get(cycle, 0) + 1
        
        # Check constraints and calculate penalties
        for staff_id, hours in staff_hours.items():
            # Monthly hours constraint (weight: 10)
            if hours['total'] > self.constraints['max_hours']:
                score += 10 * (hours['total'] - self.constraints['max_hours'])
            
            # Weekly hours constraint (weight: 8)
            for week_hours in hours['weekly'].values():
                if week_hours > 40:
                    score += 8 * (week_hours - 40)
        
        # Coverage constraints (weight: 6)
        for day in range(1, self.num_days + 1):
            day_coverage = daily_coverage.get(day, {})
            required = self.staff_requirements[day-1]
            
            for cycle, needed in required.items():
                assigned = day_coverage.get(cycle, 0)
                if assigned < needed:
                    score += 6 * (needed - assigned)
        
        # Rest days constraint (weight: 4)
        consecutive_days = {}
        for shift in schedule:
            staff_id = shift['staff_id']
            if staff_id not in consecutive_days:
                consecutive_days[staff_id] = []
            consecutive_days[staff_id].append(shift['day'])
        
        for staff_days in consecutive_days.values():
            staff_days.sort()
            for i in range(len(staff_days) - 1):
                if staff_days[i + 1] - staff_days[i] < 1:
                    score += 4
        
        # Cache the result
        self.cache[schedule_key] = score
        return score
