import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import gradio as gr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

class DialysisScheduler:
    def __init__(self, beds=12, staff_ratio=3, cycle_time=5, operation_hours=20, 
                 planning_horizon=28, max_monthly_hours=234, max_weekly_hours=54, 
                 min_gap=10, rest_days=1, shift_lengths=[12, 8, 6], min_start_gap=1):
        # Main parameters
        self.BEDS = beds
        self.STAFF_RATIO = 1/staff_ratio  # 1:3 becomes 1/3
        self.CYCLE_TIME = cycle_time
        self.OPERATION_HOURS = operation_hours
        self.PLANNING_HORIZON = planning_horizon
        self.MAX_MONTHLY_HOURS = max_monthly_hours
        self.MAX_WEEKLY_HOURS = max_weekly_hours
        self.MIN_GAP = min_gap
        self.REST_DAYS = rest_days
        self.SHIFT_LENGTHS = shift_lengths
        self.MIN_START_GAP = min_start_gap
        
        # Derived parameters
        self.CENTER_START = 6  # Assuming 6 AM start
        self.CENTER_END = self.CENTER_START + self.OPERATION_HOURS
        self.LEAD_TECH_START = 9  # 9:30 AM rounded down
        self.LEAD_TECH_DURATION = 9  # 9 hour shift (9:30 AM to 6:30 PM)
        
        # Time slots in 1-hour increments (24 hours)
        self.time_slots = range(24)
        self.days = range(self.PLANNING_HORIZON)
        
    def calculate_min_staff(self):
        # Calculate minimum staff needed based on beds and ratio
        min_staff_per_shift = np.ceil(self.BEDS * self.STAFF_RATIO)
        
        # Calculate shifts needed to cover operation hours
        shifts_per_day = np.ceil(self.OPERATION_HOURS / max(self.SHIFT_LENGTHS))
        additional_shifts = 1  # Lead technician
        
        # Account for rest days in a full cycle (planning horizon)
        rest_factor = self.PLANNING_HORIZON / (self.PLANNING_HORIZON - (self.REST_DAYS * (self.PLANNING_HORIZON / 7)))
        
        # Calculate total needed staff considering shifts, rest days, and max hours
        max_hours_per_staff = min(self.MAX_MONTHLY_HOURS, self.MAX_WEEKLY_HOURS * (self.PLANNING_HORIZON / 7))
        avg_shift_length = sum(self.SHIFT_LENGTHS) / len(self.SHIFT_LENGTHS)
        max_shifts_per_staff = max_hours_per_staff / avg_shift_length
        
        total_shifts_needed = (min_staff_per_shift * shifts_per_day + additional_shifts) * self.PLANNING_HORIZON
        min_staff_needed = np.ceil((total_shifts_needed * rest_factor) / max_shifts_per_staff)
        
        return int(min_staff_needed)
    
    def optimize_schedule(self, num_staff):
        try:
            # Create optimization model
            model = gp.Model("DialysisScheduling")
            model.setParam('OutputFlag', 0)  # Suppress output for cleaner interface
            
            staff = range(num_staff)
            
            # Decision variables: staff s works on day d starting at time t for duration dur
            x = model.addVars(staff, self.days, self.time_slots, self.SHIFT_LENGTHS, vtype=GRB.BINARY,
                             name="shift_assignment")
            
            # Objective: Minimize total working hours (minimize cost)
            model.setObjective(gp.quicksum(x[s,d,t,dur] * dur 
                                          for s in staff 
                                          for d in self.days 
                                          for t in self.time_slots 
                                          for dur in self.SHIFT_LENGTHS), 
                              GRB.MINIMIZE)
            
            # CONSTRAINTS
            
            # 1. One shift per staff per day at most
            for s in staff:
                for d in self.days:
                    model.addConstr(gp.quicksum(x[s,d,t,dur] 
                                              for t in self.time_slots 
                                              for dur in self.SHIFT_LENGTHS) <= 1)
            
            # 2. Maximum weekly hours (rolling window of 7 days)
            for s in staff:
                for start_day in range(self.PLANNING_HORIZON - 6):
                    week_days = range(start_day, start_day + 7)
                    model.addConstr(gp.quicksum(x[s,d,t,dur] * dur 
                                              for d in week_days if d < self.PLANNING_HORIZON
                                              for t in self.time_slots 
                                              for dur in self.SHIFT_LENGTHS) <= self.MAX_WEEKLY_HOURS)
            
            # 3. Maximum monthly hours
            for s in staff:
                model.addConstr(gp.quicksum(x[s,d,t,dur] * dur 
                                          for d in self.days 
                                          for t in self.time_slots 
                                          for dur in self.SHIFT_LENGTHS) <= self.MAX_MONTHLY_HOURS)
            
            # 4. Minimum gap between shifts (across days)
            for s in staff:
                for d in range(self.PLANNING_HORIZON - 1):  # All but last day
                    for t1 in self.time_slots:
                        for dur1 in self.SHIFT_LENGTHS:
                            end_time = t1 + dur1
                            
                            # Calculate hours until next day starts
                            hours_until_next_day = 24 - end_time
                            
                            # If gap extends to next day
                            if hours_until_next_day < self.MIN_GAP:
                                remaining_gap = self.MIN_GAP - hours_until_next_day
                                
                                # Prevent shifts in the beginning of next day
                                model.addConstr(x[s,d,t1,dur1] + 
                                              gp.quicksum(x[s,d+1,t2,dur2] 
                                                        for t2 in range(min(int(remaining_gap), 24))
                                                        for dur2 in self.SHIFT_LENGTHS) <= 1)
            
            # 5. At least one rest day per week
            for s in staff:
                for start_day in range(0, self.PLANNING_HORIZON - 6, 7):
                    week_days = range(start_day, min(start_day + 7, self.PLANNING_HORIZON))
                    # At least one day with no shifts
                    model.addConstr(gp.quicksum(x[s,d,t,dur] 
                                              for d in week_days 
                                              for t in self.time_slots 
                                              for dur in self.SHIFT_LENGTHS) <= 7 - self.REST_DAYS)
            
            # 6. Sufficient staff coverage during operating hours
            min_staff_needed = np.ceil(self.BEDS * self.STAFF_RATIO)
            
            for d in self.days:
                for hour in range(self.CENTER_START, self.CENTER_END):
                    # Count staff working during this hour
                    model.addConstr(
                        gp.quicksum(x[s,d,t,dur] 
                                   for s in staff
                                   for t in range(max(0, hour - max(self.SHIFT_LENGTHS) + 1), hour + 1)
                                   for dur in self.SHIFT_LENGTHS
                                   if t + dur > hour) >= min_staff_needed
                    )
            
            # 7. Lead Technician schedule (first staff member is lead tech)
            lead_tech = 0
            for d in range(min(6, self.PLANNING_HORIZON)):  # First week (6 days)
                model.addConstr(x[lead_tech, d, self.LEAD_TECH_START, self.LEAD_TECH_DURATION] == 1)
            
            # For subsequent weeks
            for week in range(1, (self.PLANNING_HORIZON // 7) + 1):
                for d in range(week * 7, min(week * 7 + 6, self.PLANNING_HORIZON)):
                    model.addConstr(x[lead_tech, d, self.LEAD_TECH_START, self.LEAD_TECH_DURATION] == 1)
            
            # 8. Minimum time between shift starts
            for d in self.days:
                for t in range(0, 24 - self.MIN_START_GAP):
                    # For all staff, no more than 1 can start in this MIN_START_GAP window
                    model.addConstr(
                        gp.quicksum(x[s,d,start_t,dur] 
                                   for s in staff
                                   for start_t in range(t, t + self.MIN_START_GAP)
                                   for dur in self.SHIFT_LENGTHS) <= min_staff_needed
                    )
            
            # Optimize the model
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                # Convert solution to schedule
                schedule = []
                for s in staff:
                    for d in self.days:
                        for t in self.time_slots:
                            for dur in self.SHIFT_LENGTHS:
                                if x[s,d,t,dur].x > 0.5:
                                    schedule.append({
                                        'Staff': f'Staff_{s}',
                                        'Day': d + 1,  # 1-indexed for readability
                                        'Start_Hour': t,
                                        'Start': f'{t:02d}:00',
                                        'Duration': dur,
                                        'End': f'{(t+dur) % 24:02d}:00',
                                        'End_Hour': (t+dur) % 24,
                                        'Hours': dur
                                    })
                
                # Calculate statistics for each staff member
                staff_stats = {}
                for s in staff:
                    staff_hours = sum(item['Hours'] for item in schedule if item['Staff'] == f'Staff_{s}')
                    staff_shifts = sum(1 for item in schedule if item['Staff'] == f'Staff_{s}')
                    staff_stats[f'Staff_{s}'] = {
                        'Total_Hours': staff_hours,
                        'Total_Shifts': staff_shifts,
                        'Avg_Hours_Per_Shift': staff_hours / staff_shifts if staff_shifts > 0 else 0
                    }
                
                return {
                    'schedule': pd.DataFrame(schedule),
                    'staff_stats': staff_stats,
                    'total_hours': sum(item['Hours'] for item in schedule),
                    'min_staff_needed': self.calculate_min_staff(),
                    'status': 'optimal'
                }
            else:
                return {
                    'status': 'infeasible',
                    'message': "No feasible schedule found with these constraints"
                }
                
        except gp.GurobiError as e:
            return {
                'status': 'error',
                'message': f"Optimization error: {str(e)}"
            }

def generate_schedule_visualization(schedule_df):
    """Generate a visual representation of the schedule"""
    if schedule_df.empty:
        return None
    
    # Group by staff and day
    staff_list = sorted(schedule_df['Staff'].unique())
    days = schedule_df['Day'].max()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(days * 0.5, len(staff_list) * 0.8))
    
    # Colors for different shift durations
    colors = {
        6: 'lightblue',
        8: 'lightgreen',
        9: 'lightyellow',  # Lead tech
        12: 'lightpink'
    }
    
    # Plot each shift as a horizontal bar
    for _, shift in schedule_df.iterrows():
        staff_idx = staff_list.index(shift['Staff'])
        start_hour = shift['Start_Hour']
        duration = shift['Duration']
        color = colors.get(duration, 'lightgray')
        
        # Day is 1-indexed in the data
        day_idx = shift['Day'] - 1
        
        # Plot the shift
        ax.barh(staff_idx, duration, left=day_idx*24 + start_hour, height=0.6, 
                color=color, edgecolor='black', alpha=0.7)
        
        # Add text for shift details
        if duration >= 4:  # Only add text if there's enough space
            text_x = day_idx*24 + start_hour + duration/2
            ax.text(text_x, staff_idx, f"{shift['Start']}-{shift['End']}", 
                    ha='center', va='center', fontsize=8)
    
    # Set y-ticks to staff names
    ax.set_yticks(range(len(staff_list)))
    ax.set_yticklabels(staff_list)
    
    # Set x-ticks to days
    day_ticks = [i*24 + 12 for i in range(days)]
    ax.set_xticks(day_ticks)
    ax.set_xticklabels([f"Day {i+1}" for i in range(days)])
    
    # Add vertical lines for day boundaries
    for i in range(1, days):
        ax.axvline(i*24, color='gray', linestyle='--', alpha=0.5)
    
    # Add horizontal lines between staff
    for i in range(1, len(staff_list)):
        ax.axhline(i - 0.5, color='gray', linestyle='-', alpha=0.2)
    
    # Add title and labels
    ax.set_title("28-Day Dialysis Center Staff Schedule")
    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("Staff")
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, color=color, label=f"{dur}h Shift") 
                       for dur, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.1), ncol=len(colors))
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Convert the figure to a base64 string
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return f'data:image/png;base64,{img_str}'

def create_html_schedule(result):
    """Create an HTML representation of the schedule"""
    if result['status'] != 'optimal':
        return f"<h3>Error: {result['message']}</h3>"
    
    schedule_df = result['schedule']
    staff_stats = result['staff_stats']
    
    # Summary statistics
    summary_html = f"""
    <div style="margin-bottom: 20px;">
        <h3>Schedule Summary</h3>
        <p><strong>Total Hours Scheduled:</strong> {result['total_hours']} hours</p>
        <p><strong>Minimum Staff Needed:</strong> {result['min_staff_needed']}</p>
        <p><strong>Staff Scheduled:</strong> {len(staff_stats)}</p>
    </div>
    """
    
    # Staff utilization table
    staff_html = """
    <div style="margin-bottom: 20px;">
        <h3>Staff Utilization</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Staff</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Total Hours</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Total Shifts</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Avg Hours/Shift</th>
            </tr>
    """
    
    for staff, stats in staff_stats.items():
        staff_html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">{staff}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{stats['Total_Hours']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{stats['Total_Shifts']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{stats['Avg_Hours_Per_Shift']:.2f}</td>
            </tr>
        """
    
    staff_html += "</table></div>"
    
    # Schedule visualization
    viz_img = generate_schedule_visualization(schedule_df)
    viz_html = f"""
    <div style="margin-bottom: 20px;">
        <h3>Schedule Visualization</h3>
        <img src="{viz_img}" style="width:100%;" alt="Schedule Visualization">
    </div>
    """ if viz_img else ""
    
    # Detailed schedule table
    schedule_html = """
    <div>
        <h3>Detailed Schedule</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Staff</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Day</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Start</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">End</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Hours</th>
            </tr>
    """
    
    # Sort schedule by staff, then day, then start time
    sorted_schedule = schedule_df.sort_values(['Staff', 'Day', 'Start_Hour'])
    
    for _, shift in sorted_schedule.iterrows():
        schedule_html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">{shift['Staff']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{shift['Day']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{shift['Start']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{shift['End']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{shift['Duration']}</td>
            </tr>
        """
    
    schedule_html += "</table></div>"
    
    return summary_html + staff_html + viz_html + schedule_html

def optimize_schedule(beds, staff_ratio, cycle_time, operation_hours, planning_horizon,
                     max_monthly_hours, max_weekly_hours, min_gap, rest_days, 
                     shift_lengths, min_start_gap, num_staff=None):
    """Optimize the dialysis center schedule with the given parameters"""
    
    # Parse shift lengths if it's a string
    if isinstance(shift_lengths, str):
        shift_lengths = [int(s.strip()) for s in shift_lengths.split(',')]
    
    # Create scheduler
    scheduler = DialysisScheduler(
        beds=beds,
        staff_ratio=staff_ratio,
        cycle_time=cycle_time,
        operation_hours=operation_hours,
        planning_horizon=planning_horizon,
        max_monthly_hours=max_monthly_hours,
        max_weekly_hours=max_weekly_hours,
        min_gap=min_gap,
        rest_days=rest_days,
        shift_lengths=shift_lengths,
        min_start_gap=min_start_gap
    )
    
    # Calculate minimum staff needed
    min_staff = scheduler.calculate_min_staff()
    
    if num_staff is None or num_staff < min_staff:
        num_staff = min_staff
        
    # Optimize
    result = scheduler.optimize_schedule(num_staff)
    
    if result['status'] == 'optimal':
        html_output = create_html_schedule(result)
        return html_output
    else:
        return f"<h3>Error: {result['message']}</h3><p>Try increasing the number of staff or adjusting constraints.</p>"

# Gradio interface
with gr.Blocks(title="Dialysis Center Staff Scheduler") as demo:
    gr.Markdown("# Advanced Healthcare Staff Scheduling Optimization")
    gr.Markdown("""
    This application optimizes staff scheduling for dialysis centers with a focus on:
    - Minimizing staff while meeting all patient needs
    - Ensuring zero overtime hours
    - Complying with all regulatory requirements
    - Accounting for Lead Technician fixed schedule
    """)
    
    with gr.Row():
        with gr.Column():
            beds = gr.Number(value=12, label="Beds per Location", min_value=1, max_value=50, step=1)
            staff_ratio = gr.Number(value=3, label="Staff to Bed Ratio (e.g., 3 for 1:3)", min_value=1, max_value=10, step=0.1)
            cycle_time = gr.Number(value=5, label="Cycle Time (hours)", min_value=1, max_value=12, step=0.5)
            operation_hours = gr.Number(value=20, label="Hours of Operation per Day", min_value=4, max_value=24, step=1)
            planning_horizon = gr.Number(value=28, label="Planning Horizon (days)", min_value=7, max_value=31, step=7)
        
        with gr.Column():
            max_monthly_hours = gr.Number(value=234, label="Maximum Monthly Hours", min_value=100, max_value=300, step=1)
            max_weekly_hours = gr.Number(value=54, label="Maximum Weekly Hours", min_value=20, max_value=80, step=1)
            min_gap = gr.Number(value=10, label="Minimum Gap Between Shifts (hours)", min_value=8, max_value=16, step=1)
            rest_days = gr.Number(value=1, label="Rest Days per Week", min_value=1, max_value=3, step=1)
            shift_lengths = gr.Textbox(value="12,8,6", label="Possible Shift Lengths (comma-separated hours)")
            min_start_gap = gr.Number(value=1, label="Minimum Start Time Gap (hours)", min_value=0, max_value=4, step=1)
    
    with gr.Row():
        num_staff = gr.Number(label="Number of Staff (leave blank for minimum)", min_value=1, step=1)
        submit_btn = gr.Button("Generate Optimal Schedule", variant="primary")
    
    output = gr.HTML(label="Schedule Output")
    
    submit_btn.click(
        fn=optimize_schedule,
        inputs=[beds, staff_ratio, cycle_time, operation_hours, planning_horizon,
               max_monthly_hours, max_weekly_hours, min_gap, rest_days, 
               shift_lengths, min_start_gap, num_staff],
        outputs=output
    )
    
    gr.Markdown("""
    ### Notes:
    - The Lead Technician is always scheduled on a stable shift (9:30 AM to 6:30 PM) for 6 days a week
    - The optimization ensures no overtime hours while minimizing the number of staff required
    - If no feasible schedule is found, try relaxing some constraints or adding more staff
    """)

if __name__ == "__main__":
    demo.launch()
