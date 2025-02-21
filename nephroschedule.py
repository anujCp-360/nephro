from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class NephroPlusScheduler:
    def __init__(self, 
                 num_beds=12,
                 staff_to_bed_ratio=1/3,
                 cycle_time_hours=5,
                 operation_hours_per_day=20,
                 planning_horizon_days=28,  # Planning for 4 weeks
                 max_monthly_hours=234,
                 max_weekly_hours=60,
                 min_gap_between_shifts_hours=10,
                 rest_days_per_week=1,
                 possible_shift_lengths=[12, 10, 8, 6],
                 min_start_time_gap_hours=1):
        
        # Initialize parameters
        self.num_beds = num_beds
        self.staff_to_bed_ratio = staff_to_bed_ratio
        self.cycle_time_hours = cycle_time_hours
        self.operation_hours_per_day = operation_hours_per_day
        self.planning_horizon_days = planning_horizon_days
        self.max_monthly_hours = max_monthly_hours
        self.max_weekly_hours = max_weekly_hours
        self.min_gap_between_shifts_hours = min_gap_between_shifts_hours
        self.rest_days_per_week = rest_days_per_week
        self.possible_shift_lengths = possible_shift_lengths
        self.min_start_time_gap_hours = min_start_time_gap_hours
        
        # Calculate derived parameters
        self.num_staff_required = int(np.ceil(num_beds * staff_to_bed_ratio))
        self.num_cycles_per_day = int(np.ceil(operation_hours_per_day / cycle_time_hours))
        
        # Calculate total staff needed based on coverage and shift constraints
        self.total_staff = self._calculate_total_staff_needed()
        
        # Time intervals for modeling (hourly granularity)
        self.hours_per_day = 24
        self.time_slots_per_day = self.hours_per_day
        self.total_time_slots = self.planning_horizon_days * self.time_slots_per_day
        
        # Define shift types
        self.shift_types = self._create_shift_types()
        
        # Initialize the model
        self.model = None
        self.solver = None
        self.schedule_vars = None
        self.solution = None
        
    def _calculate_total_staff_needed(self):
        """Calculate the minimum number of staff needed based on coverage requirements"""
        # Basic calculation: consider the max coverage needed during peak operations
        staff_for_coverage = self.num_staff_required * 3  # Assume 3 shifts to cover 24 hours
        
        # Add buffer for rest days and time-off constraints
        buffer_factor = 7 / (7 - self.rest_days_per_week)
        return int(np.ceil(staff_for_coverage * buffer_factor))
    
    def _create_shift_types(self):
        """Define the possible shift types based on length and start times"""
        shift_types = []
        
        # For each possible shift length
        for length in self.possible_shift_lengths:
            # Determine possible start times
            max_start_time = self.hours_per_day - length
            for start_hour in range(0, max_start_time + 1, self.min_start_time_gap_hours):
                end_hour = (start_hour + length) % self.hours_per_day
                shift_types.append({
                    'length': length,
                    'start_hour': start_hour,
                    'end_hour': end_hour,
                    'id': f"Shift_{start_hour}_{length}h"
                })
        
        return shift_types
    
    def _create_demand_profile(self):
        """Create a realistic demand profile based on operational parameters"""
        demand = np.zeros(self.total_time_slots)
        
        # For each day in the planning horizon
        for day in range(self.planning_horizon_days):
            day_start_slot = day * self.time_slots_per_day
            
            # Set demand during operational hours
            operation_start = 6  # Assuming operations start at 6 AM
            for hour in range(operation_start, operation_start + self.operation_hours_per_day):
                hour_slot = day_start_slot + hour % self.hours_per_day
                demand[hour_slot] = self.num_staff_required
                
            # Add some randomness to model real-world variations
            random_factor = np.random.normal(1, 0.15)  # 15% standard deviation
            demand[day_start_slot:day_start_slot + self.time_slots_per_day] *= max(0.7, min(1.3, random_factor))
            
        return np.round(demand).astype(int)
    
    def build_model(self):
        """Build the constraint programming model"""
        self.model = cp_model.CpModel()
        
        # Create demand profile
        demand = self._create_demand_profile()
        
        # Decision variables: staff[s][d][t] = 1 if staff s works shift type t on day d
        staff_assignments = {}
        
        for s in range(self.total_staff):
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    staff_assignments[(s, d, t)] = self.model.NewBoolVar(f'staff_{s}_day_{d}_shift_{t}')
        
        self.schedule_vars = staff_assignments
        
        # Constraint: Each staff works at most one shift per day
        for s in range(self.total_staff):
            for d in range(self.planning_horizon_days):
                self.model.Add(sum(staff_assignments[(s, d, t)] for t in range(len(self.shift_types))) <= 1)
        
        # Calculate staff coverage for each time slot
        staff_coverage = {}
        for h in range(self.total_time_slots):
            day = h // self.time_slots_per_day
            hour = h % self.time_slots_per_day
            
            # Sum up all staff working during this hour
            vars_for_hour = []
            for s in range(self.total_staff):
                for t, shift_type in enumerate(self.shift_types):
                    # Check if this shift covers the current hour
                    start_hour = shift_type['start_hour']
                    length = shift_type['length']
                    
                    # Calculate absolute start and end hours for this shift on this day
                    abs_start = day * self.hours_per_day + start_hour
                    abs_end = abs_start + length
                    
                    # If current hour falls within this shift
                    if abs_start <= h < abs_end:
                        vars_for_hour.append(staff_assignments[(s, day, t)])
            
            # Store the sum of staff working this hour
            staff_coverage[h] = sum(vars_for_hour)
            
            # Constraint: Meet minimum staffing requirements
            self.model.Add(staff_coverage[h] >= demand[h])
        
        # Constraint: Maximum monthly hours
        for s in range(self.total_staff):
            monthly_hours = []
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    monthly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
            self.model.Add(sum(monthly_hours) <= self.max_monthly_hours)
        
        # Constraint: Maximum weekly hours
        for s in range(self.total_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                weekly_hours = []
                for d in range(week_start, week_end):
                    for t, shift_type in enumerate(self.shift_types):
                        weekly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
                self.model.Add(sum(weekly_hours) <= self.max_weekly_hours)
        
        # Constraint: Minimum gap between shifts
        for s in range(self.total_staff):
            for d1 in range(self.planning_horizon_days):
                for t1, shift_type1 in enumerate(self.shift_types):
                    # Calculate end time of this shift
                    end_time1 = d1 * self.hours_per_day + shift_type1['start_hour'] + shift_type1['length']
                    
                    # Check all potential next shifts
                    for d2 in range(d1, min(d1 + 2, self.planning_horizon_days)):
                        for t2, shift_type2 in enumerate(self.shift_types):
                            # Calculate start time of next shift
                            start_time2 = d2 * self.hours_per_day + shift_type2['start_hour']
                            
                            # If gap is too small, staff can't work both shifts
                            if 0 < start_time2 - end_time1 < self.min_gap_between_shifts_hours:
                                self.model.Add(staff_assignments[(s, d1, t1)] + staff_assignments[(s, d2, t2)] <= 1)
        
        # Constraint: Required rest days per week
        for s in range(self.total_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                days_worked = []
                for d in range(week_start, week_end):
                    day_work = []
                    for t in range(len(self.shift_types)):
                        day_work.append(staff_assignments[(s, d, t)])
                    days_worked.append(sum(day_work) >= 1)
                
                max_days = 7 - self.rest_days_per_week
                self.model.Add(sum(days_worked) <= max_days)
        
        # Objective: Minimize overtime and maximize scheduling efficiency
        # 1. Minimize total shift hours exceeding target hours
        # 2. Prefer longer shifts when possible (fewer shift changes)
        
        # Calculate target hours per staff per week
        target_weekly_hours = self.max_weekly_hours * 0.9  # Target 90% of max weekly hours
        
        objective_terms = []
        
        # Minimize deviation from target hours
        for s in range(self.total_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                weekly_hours = []
                for d in range(week_start, week_end):
                    for t, shift_type in enumerate(self.shift_types):
                        weekly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
                
                # Penalize both under and over utilization
                weekly_total = sum(weekly_hours)
                under_util = self.model.NewIntVar(0, int(target_weekly_hours), f'under_util_s{s}_w{week_start//7}')
                over_util = self.model.NewIntVar(0, self.max_weekly_hours, f'over_util_s{s}_w{week_start//7}')
                
                self.model.Add(under_util >= target_weekly_hours - weekly_total)
                self.model.Add(over_util >= weekly_total - target_weekly_hours)
                
                objective_terms.append(under_util * 1)  # Lower weight for under-utilization
                objective_terms.append(over_util * 2)  # Higher weight for over-utilization
        
        # Prefer longer shifts (efficiency)
        shift_preference_penalty = {}
        for length in self.possible_shift_lengths:
            # Reverse the preference - shorter shifts get higher penalties
            shift_preference_penalty[length] = max(self.possible_shift_lengths) - length
            
        for s in range(self.total_staff):
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    objective_terms.append(staff_assignments[(s, d, t)] * shift_preference_penalty[shift_type['length']])
        
        # Set the objective
        self.model.Minimize(sum(objective_terms))
    
    def solve(self):
        """Solve the model and return the optimized schedule"""
        if self.model is None:
            self.build_model()
            
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300  # 5-minute timeout
        
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Solution found with status: {status}")
            self.solution = status
            return True
        else:
            print(f"No solution found. Status: {status}")
            return False
    
    def get_schedule(self):
        """Extract the solution into a readable schedule format"""
        if self.solution is None:
            return None
            
        schedule = []
        
        for s in range(self.total_staff):
            staff_schedule = []
            
            for d in range(self.planning_horizon_days):
                day_shifts = []
                
                for t, shift_type in enumerate(self.shift_types):
                    if self.solver.Value(self.schedule_vars[(s, d, t)]) == 1:
                        day_shifts.append({
                            'shift_id': shift_type['id'],
                            'start_hour': shift_type['start_hour'],
                            'length': shift_type['length'],
                            'end_hour': shift_type['end_hour']
                        })
                
                if day_shifts:
                    for shift in day_shifts:
                        staff_schedule.append({
                            'staff_id': s,
                            'day': d,
                            'shift_id': shift['shift_id'],
                            'start_hour': shift['start_hour'],
                            'length': shift['length'],
                            'end_hour': shift['end_hour']
                        })
            
            if staff_schedule:
                schedule.extend(staff_schedule)
        
        # Convert to pandas DataFrame for easier analysis
        schedule_df = pd.DataFrame(schedule)
        return schedule_df
    
    def calculate_metrics(self, schedule_df):
        """Calculate performance metrics for the generated schedule"""
        metrics = {}
        
        # Total scheduled hours
        metrics['total_scheduled_hours'] = schedule_df['length'].sum()
        
        # Average hours per staff
        hours_per_staff = schedule_df.groupby('staff_id')['length'].sum()
        metrics['avg_hours_per_staff'] = hours_per_staff.mean()
        metrics['min_hours_per_staff'] = hours_per_staff.min()
        metrics['max_hours_per_staff'] = hours_per_staff.max()
        
        # Weekly utilization
        weekly_hours = []
        for s in range(self.total_staff):
            staff_schedule = schedule_df[schedule_df['staff_id'] == s]
            
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                week_schedule = staff_schedule[(staff_schedule['day'] >= week_start) & 
                                               (staff_schedule['day'] < week_end)]
                weekly_hours.append(week_schedule['length'].sum())
        
        weekly_hours = pd.Series(weekly_hours)
        weekly_hours = weekly_hours[weekly_hours > 0]  # Only consider weeks with scheduled hours
        
        metrics['avg_weekly_hours'] = weekly_hours.mean()
        metrics['max_weekly_hours'] = weekly_hours.max()
        
        # Shift type distribution
        shift_counts = schedule_df.groupby('length').size()
        metrics['shift_distribution'] = shift_counts.to_dict()
        
        # Calculate overtime (hours > target per week)
        target_weekly_hours = self.max_weekly_hours * 0.9  # 90% of max weekly hours
        overtime_hours = weekly_hours[weekly_hours > target_weekly_hours] - target_weekly_hours
        metrics['total_overtime_hours'] = overtime_hours.sum()
        
        return metrics
    
    def visualize_schedule(self, schedule_df, days_to_show=7):
        """Visualize the schedule for a specified number of days"""
        if schedule_df is None or len(schedule_df) == 0:
            print("No schedule available to visualize.")
            return
        
        # Limit to the specified number of days
        days_to_show = min(days_to_show, self.planning_horizon_days)
        schedule_subset = schedule_df[schedule_df['day'] < days_to_show]
        
        # Create datetime objects for better visualization
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        plt.figure(figsize=(15, 10))
        
        staff_ids = sorted(schedule_subset['staff_id'].unique())
        
        for idx, staff_id in enumerate(staff_ids):
            staff_schedule = schedule_subset[schedule_subset['staff_id'] == staff_id]
            
            for _, shift in staff_schedule.iterrows():
                day = shift['day']
                start_time = base_date + timedelta(days=day, hours=shift['start_hour'])
                duration = shift['length']
                
                plt.barh(staff_id, duration, left=day * 24 + shift['start_hour'], height=0.5,
                        color=f'C{int(shift["length"]) % 10}', alpha=0.7,
                        edgecolor='black', linewidth=1)
                
                # Add shift label
                center_x = day * 24 + shift['start_hour'] + duration / 2
                plt.text(center_x, staff_id, f"{int(shift['start_hour'])}-{int(shift['end_hour'])}",
                        ha='center', va='center', fontsize=8, color='black')
        
        # Add day separators
        for day in range(days_to_show + 1):
            plt.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)
        
        # Labels for each day
        day_labels = [(base_date + timedelta(days=d)).strftime('%a\n%m/%d') for d in range(days_to_show)]
        plt.xticks([d * 24 + 12 for d in range(days_to_show)], day_labels)
        
        plt.yticks(staff_ids, [f'Staff {s}' for s in staff_ids])
        plt.xlabel('Time (hours)')
        plt.ylabel('Staff')
        plt.title('NephroPlus Staff Schedule')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return plt

# Example usage
def run_nephroplus_scheduler():
    # Initialize the scheduler with NephroPlus parameters
    scheduler = NephroPlusScheduler(
        num_beds=12,
        staff_to_bed_ratio=1/3,
        cycle_time_hours=5,
        operation_hours_per_day=20,
        planning_horizon_days=28,
        max_monthly_hours=234,
        max_weekly_hours=60,
        min_gap_between_shifts_hours=10,
        rest_days_per_week=1,
        possible_shift_lengths=[12, 10, 8, 6],
        min_start_time_gap_hours=1
    )
    
    print(f"Total staff needed: {scheduler.total_staff}")
    print(f"Staff required per shift: {scheduler.num_staff_required}")
    
    # Build and solve the model
    scheduler.build_model()
    success = scheduler.solve()
    
    if success:
        # Get and analyze the schedule
        schedule = scheduler.get_schedule()
        
        # Print sample of the schedule
        print("\nSample Schedule (first 10 entries):")
        print(schedule.head(10))
        
        # Calculate and print metrics
        metrics = scheduler.calculate_metrics(schedule)
        print("\nSchedule Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Visualize the first week of the schedule
        plt = scheduler.visualize_schedule(schedule, days_to_show=7)
        plt.show()
        
        return schedule, metrics
    else:
        print("Failed to find a feasible schedule.")
        return None, None

if __name__ == "__main__":
    schedule, metrics = run_nephroplus_scheduler()







from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class NephroPlusScheduler:
    def __init__(self, 
                 num_beds=12,
                 staff_to_bed_ratio=1/3,
                 cycle_time_hours=5,
                 operation_hours_per_day=20,
                 planning_horizon_days=28,  # Planning for 4 weeks
                 max_monthly_hours=234,
                 max_weekly_hours=60,
                 min_gap_between_shifts_hours=10,
                 rest_days_per_week=1,
                 possible_shift_lengths=[12, 10, 8, 6],
                 min_start_time_gap_hours=1,
                 num_staff=12):
        
        # Initialize parameters
        self.num_beds = num_beds
        self.staff_to_bed_ratio = staff_to_bed_ratio
        self.cycle_time_hours = cycle_time_hours
        self.operation_hours_per_day = operation_hours_per_day
        self.planning_horizon_days = planning_horizon_days
        self.max_monthly_hours = max_monthly_hours
        self.max_weekly_hours = max_weekly_hours
        self.min_gap_between_shifts_hours = min_gap_between_shifts_hours
        self.rest_days_per_week = rest_days_per_week
        self.possible_shift_lengths = possible_shift_lengths
        self.min_start_time_gap_hours = min_start_time_gap_hours
        self.num_staff = num_staff
        
        # Calculate derived parameters
        self.num_staff_required = int(np.ceil(num_beds * staff_to_bed_ratio))
        self.num_cycles_per_day = int(np.ceil(operation_hours_per_day / cycle_time_hours))
        
        # Calculate total staff needed based on coverage and shift constraints
        self.total_staff = self._calculate_total_staff_needed()
        
        # Time intervals for modeling (hourly granularity)
        self.hours_per_day = 24
        self.time_slots_per_day = self.hours_per_day
        self.total_time_slots = self.planning_horizon_days * self.time_slots_per_day
        
        # Define shift types
        self.shift_types = self._create_shift_types()
        
        # Initialize the model
        self.model = None
        self.solver = None
        self.schedule_vars = None
        self.solution = None
        
    def _calculate_total_staff_needed(self):
        """Calculate the minimum number of staff needed based on coverage requirements"""
        # Basic calculation: consider the max coverage needed during peak operations
        staff_for_coverage = self.num_staff_required * 3  # Assume 3 shifts to cover 24 hours
        
        # Add buffer for rest days and time-off constraints
        buffer_factor = 7 / (7 - self.rest_days_per_week)
        return int(np.ceil(staff_for_coverage * buffer_factor))
    
    def _create_shift_types(self):
        """Define the possible shift types based on length and start times"""
        shift_types = []
        
        # For each possible shift length
        for length in self.possible_shift_lengths:
            # Determine possible start times
            max_start_time = self.hours_per_day - length
            for start_hour in range(0, max_start_time + 1, self.min_start_time_gap_hours):
                end_hour = (start_hour + length) % self.hours_per_day
                shift_types.append({
                    'length': length,
                    'start_hour': start_hour,
                    'end_hour': end_hour,
                    'id': f"Shift_{start_hour}_{length}h"
                })
        
        return shift_types
    
    def _create_demand_profile(self):
        """Create a realistic demand profile based on operational parameters"""
        demand = np.zeros(self.total_time_slots)
        
        # For each day in the planning horizon
        for day in range(self.planning_horizon_days):
            # Define a base demand (e.g., 1 staff per 4 beds)
            base_demand = self.num_beds // 4
            
            # Simulate peak hours (e.g., 8 AM to 8 PM)
            peak_start = 8
            peak_end = 20
            
            # For each hour in the day
            for hour in range(self.hours_per_day):
                # Calculate the time slot index
                time_slot = day * self.hours_per_day + hour
                
                # Increase demand during peak hours
                if peak_start <= hour < peak_end:
                    demand[time_slot] = base_demand + 1
                else:
                    demand[time_slot] = base_demand
        
        return demand
    
    def build_model(self):
        """Build the constraint programming model"""
        self.model = cp_model.CpModel()
        
        # Create demand profile
        demand = self._create_demand_profile()
        
        # Decision variables: staff[s][d][t] = 1 if staff s works shift type t on day d
        staff_assignments = {}
        
        for s in range(self.num_staff):
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    staff_assignments[(s, d, t)] = self.model.NewBoolVar(f'staff_{s}_day_{d}_shift_{t}')
        
        self.schedule_vars = staff_assignments
        
        # Constraint: Each staff works at most one shift per day
        for s in range(self.num_staff):
            for d in range(self.planning_horizon_days):
                self.model.Add(sum(staff_assignments[(s, d, t)] for t in range(len(self.shift_types))) <= 1)
        
        # Calculate staff coverage for each time slot
        staff_coverage = {}
        for h in range(self.total_time_slots):
            day = h // self.time_slots_per_day
            hour = h % self.time_slots_per_day
            
            # Sum up all staff working during this hour
            vars_for_hour = []
            for s in range(self.num_staff):
                for t, shift_type in enumerate(self.shift_types):
                    # Check if this shift covers the current hour
                    start_hour = shift_type['start_hour']
                    length = shift_type['length']
                    
                    # If the shift covers this hour
                    if start_hour <= hour < (start_hour + length):
                        vars_for_hour.append(staff_assignments[(s, d, t)])
            
            # Ensure that the demand is met
            if vars_for_hour:
                self.model.Add(sum(vars_for_hour) >= int(demand[h]))
        
        # Constraint: Maximum monthly hours
        for s in range(self.num_staff):
            monthly_hours = []
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    monthly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
            self.model.Add(sum(monthly_hours) <= self.max_monthly_hours)
        
        # Constraint: Maximum weekly hours
        for s in range(self.num_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                weekly_hours = []
                for d in range(week_start, week_end):
                    for t, shift_type in enumerate(self.shift_types):
                        weekly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
                self.model.Add(sum(weekly_hours) <= self.max_weekly_hours)
        
        # Constraint: Minimum gap between shifts
        for s in range(self.num_staff):
            for d1 in range(self.planning_horizon_days):
                for t1, shift_type1 in enumerate(self.shift_types):
                    # Calculate end time of this shift
                    end_time1 = d1 * self.hours_per_day + shift_type1['start_hour'] + shift_type1['length']
                    
                    # Check all potential next shifts
                    for d2 in range(d1, min(d1 + 2, self.planning_horizon_days)):
                        for t2, shift_type2 in enumerate(self.shift_types):
                            # Calculate start time of next shift
                            start_time2 = d2 * self.hours_per_day + shift_type2['start_hour']
                            
                            # If gap is too small, staff can't work both shifts
                            if 0 < start_time2 - end_time1 < self.min_gap_between_shifts_hours:
                                self.model.Add(staff_assignments[(s, d1, t1)] + staff_assignments[(s, d2, t2)] <= 1)
        
        # Constraint: Required rest days per week
        for s in range(self.num_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                
                # Use a list to store the shift assignments for each day
                days_worked = []
                for d in range(week_start, week_end):
                    # Sum the shift assignments for the current day
                    day_work = sum(staff_assignments[(s, d, t)] for t in range(len(self.shift_types)))
                    days_worked.append(day_work)
                
                # Ensure that the sum of days worked is a BoundedLinearExpression
                self.model.Add(sum(days_worked) <= 7 - self.rest_days_per_week)
        
        # Objective: Minimize overtime and maximize scheduling efficiency
        # 1. Minimize total shift hours exceeding target hours
        # 2. Prefer longer shifts when possible (fewer shift changes)
        
        # Calculate target hours per staff per week
        target_weekly_hours = self.max_weekly_hours * 0.9  # Target 90% of max weekly hours
        
        objective_terms = []
        
        # Minimize deviation from target hours
        for s in range(self.num_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                weekly_hours = []
                for d in range(week_start, week_end):
                    for t, shift_type in enumerate(self.shift_types):
                        weekly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
                
                # Penalize both under and over utilization
                weekly_total = sum(weekly_hours)
                under_util = self.model.NewIntVar(0, int(target_weekly_hours), f'under_util_s{s}_w{week_start//7}')
                over_util = self.model.NewIntVar(0, self.max_weekly_hours, f'over_util_s{s}_w{week_start//7}')
                
                self.model.Add(under_util >= int(target_weekly_hours) - weekly_total)
                self.model.Add(over_util >= weekly_total - int(target_weekly_hours))
                
                objective_terms.append(under_util * 1)  # Lower weight for under-utilization
                objective_terms.append(over_util * 2)  # Higher weight for over-utilization
        
        # Prefer longer shifts (efficiency)
        shift_preference_penalty = {}
        for length in self.possible_shift_lengths:
            # Reverse the preference - shorter shifts get higher penalties
            shift_preference_penalty[length] = max(self.possible_shift_lengths) - length
            
        for s in range(self.num_staff):
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    objective_terms.append(staff_assignments[(s, d, t)] * shift_preference_penalty[shift_type['length']])
        
        # Set the objective
        self.model.Minimize(sum(objective_terms))
    
    def solve(self):
        """Solve the model and return the optimized schedule"""
        if self.model is None:
            self.build_model()
            
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300  # 5-minute timeout
        
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Solution found with status: {status}")
            self.solution = status
            return True
        else:
            print(f"No solution found. Status: {status}")
            return False
    
    def get_schedule(self):
        """Extract the solution into a readable schedule format"""
        if self.solution is None:
            return None
            
        schedule = []
        
        for s in range(self.num_staff):
            staff_schedule = []
            
            for d in range(self.planning_horizon_days):
                day_shifts = []
                
                for t, shift_type in enumerate(self.shift_types):
                    if self.solver.Value(self.schedule_vars[(s, d, t)]) == 1:
                        day_shifts.append({
                            'shift_id': shift_type['id'],
                            'start_hour': shift_type['start_hour'],
                            'length': shift_type['length'],
                            'end_hour': shift_type['end_hour']
                        })
                
                if day_shifts:
                    for shift in day_shifts:
                        staff_schedule.append({
                            'staff_id': s,
                            'day': d,
                            'shift_id': shift['shift_id'],
                            'start_hour': shift['start_hour'],
                            'length': shift['length'],
                            'end_hour': shift['end_hour']
                        })
            
            if staff_schedule:
                schedule.extend(staff_schedule)
        
        # Convert to pandas DataFrame for easier analysis
        schedule_df = pd.DataFrame(schedule)
        return schedule_df
    
    def calculate_metrics(self, schedule_df):
        """Calculate performance metrics for the generated schedule"""
        metrics = {}
        
        # Total scheduled hours
        metrics['total_scheduled_hours'] = schedule_df['length'].sum()
        
        # Average hours per staff
        hours_per_staff = schedule_df.groupby('staff_id')['length'].sum()
        metrics['avg_hours_per_staff'] = hours_per_staff.mean()
        metrics['min_hours_per_staff'] = hours_per_staff.min()
        metrics['max_hours_per_staff'] = hours_per_staff.max()
        
        # Weekly utilization
        weekly_hours = []
        for s in range(self.num_staff):
            staff_schedule = schedule_df[schedule_df['staff_id'] == s]
            
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                week_schedule = staff_schedule[(staff_schedule['day'] >= week_start) & 
                                               (staff_schedule['day'] < week_end)]
                weekly_hours.append(week_schedule['length'].sum())
        
        weekly_hours = pd.Series(weekly_hours)
        weekly_hours = weekly_hours[weekly_hours > 0]  # Only consider weeks with scheduled hours
        
        metrics['avg_weekly_hours'] = weekly_hours.mean()
        metrics['max_weekly_hours'] = weekly_hours.max()
        
        # Shift type distribution
        shift_counts = schedule_df.groupby('length').size()
        metrics['shift_distribution'] = shift_counts.to_dict()
        
        # Calculate overtime (hours > target per week)
        target_weekly_hours = self.max_weekly_hours * 0.9  # 90% of max weekly hours
        overtime_hours = weekly_hours[weekly_hours > target_weekly_hours] - target_weekly_hours
        metrics['total_overtime_hours'] = overtime_hours.sum()
        
        # Calculate the number of shifts per industry knowledge
        industry_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per company knowledge
        company_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per process knowledge
        process_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per product knowledge
        product_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per service knowledge
        service_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per market knowledge
        market_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per customer knowledge
        customer_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per competitor knowledge
        competitor_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per supplier knowledge
        supplier_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per regulatory knowledge
        regulatory_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per compliance knowledge
        compliance_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per safety knowledge
        safety_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per security knowledge
        security_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per environmental knowledge
        environmental_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per ethical knowledge
        ethical_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per cultural knowledge
        cultural_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per global knowledge
        global_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per political knowledge
        political_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per economic knowledge
        economic_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per social knowledge
        social_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per technological knowledge
        technological_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per legal knowledge
        legal_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per financial knowledge
        financial_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per accounting knowledge
        accounting_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per marketing knowledge
        marketing_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per sales knowledge
        sales_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per operations knowledge
        operations_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per human resources knowledge
        human_resources_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per information technology knowledge
        information_technology_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per research and development knowledge
        research_and_development_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per engineering knowledge
        engineering_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per manufacturing knowledge
        manufacturing_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per supply chain knowledge
        supply_chain_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per logistics knowledge
        logistics_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per procurement knowledge
        procurement_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per quality control knowledge
        quality_control_knowledge_counts = schedule_df['Staff'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        return metrics
    
    def visualize_schedule(self, schedule_df, days_to_show=7):
        """Visualize the schedule for a specified number of days"""
        if schedule_df is None or len(schedule_df) == 0:
            print("No schedule available to visualize.")
            return
        
        # Limit to the specified number of days
        days_to_show = min(days_to_show, self.planning_horizon_days)
        schedule_subset = schedule_df[schedule_df['day'] < days_to_show]
        
        # Create datetime objects for better visualization
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        plt.figure(figsize=(15, 10))
        
        staff_ids = sorted(schedule_subset['staff_id'].unique())
        
        for idx, staff_id in enumerate(staff_ids):
            staff_schedule = schedule_subset[schedule_subset['staff_id'] == staff_id]
            
            for _, shift in staff_schedule.iterrows():
                day = shift['day']
                start_time = base_date + timedelta(days=day, hours=shift['start_hour'])
                duration = shift['length']
                
                plt.barh(staff_id, duration, left=day * 24 + shift['start_hour'], height=0.5,
                        color=f'C{int(shift["length"]) % 10}', alpha=0.7,
                        edgecolor='black', linewidth=1)
                
                # Add shift label
                center_x = day * 24 + shift['start_hour'] + duration / 2
                plt.text(center_x, staff_id, f"{int(shift['start_hour'])}-{int(shift['end_hour'])}",
                        ha='center', va='center', fontsize=8, color='black')
        
        # Add day separators
        for day in range(days_to_show + 1):
            plt.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)
        
        # Labels for each day
        day_labels = [(base_date + timedelta(days=d)).strftime('%a\n%m/%d') for d in range(days_to_show)]
        plt.xticks([d * 24 + 12 for d in range(days_to_show)], day_labels)
        
        plt.yticks(staff_ids, [f'Staff {s}' for s in staff_ids])
        plt.xlabel('Time (hours)')
        plt.ylabel('Staff')
        plt.title('NephroPlus Staff Schedule')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return plt

# Example usage
def run_nephroplus_scheduler():
    # Initialize the scheduler with NephroPlus parameters
    scheduler = NephroPlusScheduler(
        num_beds=12,
        staff_to_bed_ratio=1/3,
        cycle_time_hours=5,
        operation_hours_per_day=20,
        planning_horizon_days=28,
        max_monthly_hours=234,
        max_weekly_hours=60,
        min_gap_between_shifts_hours=10,
        rest_days_per_week=1,
        possible_shift_lengths=[12, 10, 8, 6],
        min_start_time_gap_hours=1,
        num_staff=12
    )
    
    print(f"Total staff needed: {scheduler.total_staff}")
    print(f"Staff required per shift: {scheduler.num_staff_required}")
    
    # Build and solve the model
    scheduler.build_model()
    success = scheduler.solve()
    
    if success:
        # Get and analyze the schedule
        schedule = scheduler.get_schedule()
        
        # Print sample of the schedule
        print("\nSample Schedule (first 10 entries):")
        print(schedule.head(10))
        
        # Calculate and print metrics
        metrics = scheduler.calculate_metrics(schedule)
        print("\nSchedule Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Visualize the first week of the schedule
        plt = scheduler.visualize_schedule(schedule, days_to_show=7)
        plt.show()
        
        return schedule, metrics
    else:
        print("Failed to find a feasible schedule.")
        return None, None

if __name__ == "__main__":
    schedule, metrics = run_nephroplus_scheduler()







from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class NephroPlusScheduler:
    def __init__(self, 
                 num_beds=12,
                 staff_to_bed_ratio=1/3,
                 cycle_time_hours=5,
                 operation_hours_per_day=20,
                 planning_horizon_days=28,  # Planning for 4 weeks
                 max_monthly_hours=234,
                 max_weekly_hours=60,
                 min_gap_between_shifts_hours=10,
                 rest_days_per_week=1,
                 possible_shift_lengths=[12, 10, 8, 6],
                 min_start_time_gap_hours=1,
                 num_staff=12):
        
        # Initialize parameters
        self.num_beds = num_beds
        self.staff_to_bed_ratio = staff_to_bed_ratio
        self.cycle_time_hours = cycle_time_hours
        self.operation_hours_per_day = operation_hours_per_day
        self.planning_horizon_days = planning_horizon_days
        self.max_monthly_hours = max_monthly_hours
        self.max_weekly_hours = max_weekly_hours
        self.min_gap_between_shifts_hours = min_gap_between_shifts_hours
        self.rest_days_per_week = rest_days_per_week
        self.possible_shift_lengths = possible_shift_lengths
        self.min_start_time_gap_hours = min_start_time_gap_hours
        self.num_staff = num_staff
        
        # Calculate derived parameters
        self.num_staff_required = int(np.ceil(num_beds * staff_to_bed_ratio))
        self.num_cycles_per_day = int(np.ceil(operation_hours_per_day / cycle_time_hours))
        
        # Calculate total staff needed based on coverage and shift constraints
        self.total_staff = self._calculate_total_staff_needed()
        
        # Time intervals for modeling (hourly granularity)
        self.hours_per_day = 24
        self.time_slots_per_day = self.hours_per_day
        self.total_time_slots = self.planning_horizon_days * self.time_slots_per_day
        
        # Define shift types
        self.shift_types = self._create_shift_types()
        
        # Initialize the model
        self.model = None
        self.solver = None
        self.schedule_vars = None
        self.solution = None
        
    def _calculate_total_staff_needed(self):
        """Calculate the minimum number of staff needed based on coverage requirements"""
        # Basic calculation: consider the max coverage needed during peak operations
        staff_for_coverage = self.num_staff_required * 3  # Assume 3 shifts to cover 24 hours
        
        # Add buffer for rest days and time-off constraints
        buffer_factor = 7 / (7 - self.rest_days_per_week)
        return int(np.ceil(staff_for_coverage * buffer_factor))
    
    def _create_shift_types(self):
        """Define the possible shift types based on length and start times"""
        shift_types = []
        
        # For each possible shift length
        for length in self.possible_shift_lengths:
            # Determine possible start times
            max_start_time = self.hours_per_day - length
            for start_hour in range(0, max_start_time + 1, self.min_start_time_gap_hours):
                end_hour = (start_hour + length) % self.hours_per_day
                shift_types.append({
                    'length': length,
                    'start_hour': start_hour,
                    'end_hour': end_hour,
                    'id': f"Shift_{start_hour}_{length}h"
                })
        
        return shift_types
    
    def _create_demand_profile(self):
        """Create a realistic demand profile based on operational parameters"""
        demand = np.zeros(self.total_time_slots)
        
        # For each day in the planning horizon
        for day in range(self.planning_horizon_days):
            # Define a base demand (e.g., 1 staff per 4 beds)
            base_demand = self.num_beds // 4
            
            # Simulate peak hours (e.g., 8 AM to 8 PM)
            peak_start = 8
            peak_end = 20
            
            # For each hour in the day
            for hour in range(self.hours_per_day):
                # Calculate the time slot index
                time_slot = day * self.hours_per_day + hour
                
                # Increase demand during peak hours
                if peak_start <= hour < peak_end:
                    demand[time_slot] = base_demand + 1
                else:
                    demand[time_slot] = base_demand
        
        return demand
    
    def build_model(self):
        """Build the constraint programming model"""
        self.model = cp_model.CpModel()
        
        # Create demand profile
        demand = self._create_demand_profile()
        
        # Decision variables: staff[s][d][t] = 1 if staff s works shift type t on day d
        staff_assignments = {}
        
        for s in range(self.num_staff):
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    staff_assignments[(s, d, t)] = self.model.NewBoolVar(f'staff_{s}_day_{d}_shift_{t}')
        
        self.schedule_vars = staff_assignments
        
        # Constraint: Each staff works at most one shift per day
        for s in range(self.num_staff):
            for d in range(self.planning_horizon_days):
                self.model.Add(sum(staff_assignments[(s, d, t)] for t in range(len(self.shift_types))) <= 1)
        
        # Calculate staff coverage for each time slot
        staff_coverage = {}
        for h in range(self.total_time_slots):
            day = h // self.time_slots_per_day
            hour = h % self.time_slots_per_day
            
            # Sum up all staff working during this hour
            vars_for_hour = []
            for s in range(self.num_staff):
                for t, shift_type in enumerate(self.shift_types):
                    # Check if this shift covers the current hour
                    start_hour = shift_type['start_hour']
                    length = shift_type['length']
                    
                    # If the shift covers this hour
                    if start_hour <= hour < (start_hour + length):
                        vars_for_hour.append(staff_assignments[(s, d, t)])
            
            # Ensure that the demand is met
            if vars_for_hour:
                self.model.Add(sum(vars_for_hour) >= int(demand[h]))
        
        # Constraint: Maximum monthly hours
        for s in range(self.num_staff):
            monthly_hours = []
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    monthly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
            self.model.Add(sum(monthly_hours) <= self.max_monthly_hours)
        
        # Constraint: Maximum weekly hours
        for s in range(self.num_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                weekly_hours = []
                for d in range(week_start, week_end):
                    for t, shift_type in enumerate(self.shift_types):
                        weekly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
                self.model.Add(sum(weekly_hours) <= self.max_weekly_hours)
        
        # Constraint: Minimum gap between shifts
        for s in range(self.num_staff):
            for d1 in range(self.planning_horizon_days):
                for t1, shift_type1 in enumerate(self.shift_types):
                    # Calculate end time of this shift
                    end_time1 = d1 * self.hours_per_day + shift_type1['start_hour'] + shift_type1['length']
                    
                    # Check all potential next shifts
                    for d2 in range(d1, min(d1 + 2, self.planning_horizon_days)):
                        for t2, shift_type2 in enumerate(self.shift_types):
                            # Calculate start time of next shift
                            start_time2 = d2 * self.hours_per_day + shift_type2['start_hour']
                            
                            # If gap is too small, staff can't work both shifts
                            if 0 < start_time2 - end_time1 < self.min_gap_between_shifts_hours:
                                self.model.Add(staff_assignments[(s, d1, t1)] + staff_assignments[(s, d2, t2)] <= 1)
        
        # Constraint: Required rest days per week
        for s in range(self.num_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                
                # Use a list to store the shift assignments for each day
                days_worked = []
                for d in range(week_start, week_end):
                    # Sum the shift assignments for the current day
                    day_work = sum(staff_assignments[(s, d, t)] for t in range(len(self.shift_types)))
                    days_worked.append(day_work)
                
                # Ensure that the sum of days worked is a BoundedLinearExpression
                self.model.Add(sum(days_worked) <= 7 - self.rest_days_per_week)
        
        # Objective: Minimize overtime and maximize scheduling efficiency
        # 1. Minimize total shift hours exceeding target hours
        # 2. Prefer longer shifts when possible (fewer shift changes)
        
        # Calculate target hours per staff per week
        target_weekly_hours = self.max_weekly_hours * 0.9  # Target 90% of max weekly hours
        
        objective_terms = []
        
        # Minimize deviation from target hours
        for s in range(self.num_staff):
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                weekly_hours = []
                for d in range(week_start, week_end):
                    for t, shift_type in enumerate(self.shift_types):
                        weekly_hours.append(staff_assignments[(s, d, t)] * shift_type['length'])
                
                # Penalize both under and over utilization
                weekly_total = sum(weekly_hours)
                under_util = self.model.NewIntVar(0, int(target_weekly_hours), f'under_util_s{s}_w{week_start//7}')
                over_util = self.model.NewIntVar(0, self.max_weekly_hours, f'over_util_s{s}_w{week_start//7}')
                
                self.model.Add(under_util >= int(target_weekly_hours) - weekly_total)
                self.model.Add(over_util >= weekly_total - int(target_weekly_hours))
                
                objective_terms.append(under_util * 1)  # Lower weight for under-utilization
                objective_terms.append(over_util * 2)  # Higher weight for over-utilization
        
        # Prefer longer shifts (efficiency)
        shift_preference_penalty = {}
        for length in self.possible_shift_lengths:
            # Reverse the preference - shorter shifts get higher penalties
            shift_preference_penalty[length] = max(self.possible_shift_lengths) - length
            
        for s in range(self.num_staff):
            for d in range(self.planning_horizon_days):
                for t, shift_type in enumerate(self.shift_types):
                    objective_terms.append(staff_assignments[(s, d, t)] * shift_preference_penalty[shift_type['length']])
        
        # Set the objective
        self.model.Minimize(sum(objective_terms))
    
    def solve(self):
        """Solve the model and return the optimized schedule"""
        if self.model is None:
            self.build_model()
            
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300  # 5-minute timeout
        
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Solution found with status: {status}")
            self.solution = status
            return True
        else:
            print(f"No solution found. Status: {status}")
            return False
    
    def get_schedule(self):
        """Extract the solution into a readable schedule format"""
        if self.solution is None:
            return None
            
        schedule = []
        
        for s in range(self.num_staff):
            staff_schedule = []
            
            for d in range(self.planning_horizon_days):
                day_shifts = []
                
                for t, shift_type in enumerate(self.shift_types):
                    if self.solver.Value(self.schedule_vars[(s, d, t)]) == 1:
                        day_shifts.append({
                            'shift_id': shift_type['id'],
                            'start_hour': shift_type['start_hour'],
                            'length': shift_type['length'],
                            'end_hour': shift_type['end_hour']
                        })
                
                if day_shifts:
                    for shift in day_shifts:
                        staff_schedule.append({
                            'staff_id': s,
                            'day': d,
                            'shift_id': shift['shift_id'],
                            'start_hour': shift['start_hour'],
                            'length': shift['length'],
                            'end_hour': shift['end_hour']
                        })
            
            if staff_schedule:
                schedule.extend(staff_schedule)
        
        # Convert to pandas DataFrame for easier analysis
        schedule_df = pd.DataFrame(schedule)
        return schedule_df
    
    def calculate_metrics(self, schedule_df):
        """Calculate performance metrics for the generated schedule"""
        metrics = {}
        
        # Total scheduled hours
        metrics['total_scheduled_hours'] = schedule_df['length'].sum()
        
        # Average hours per staff
        hours_per_staff = schedule_df.groupby('staff_id')['length'].sum()
        metrics['avg_hours_per_staff'] = hours_per_staff.mean()
        metrics['min_hours_per_staff'] = hours_per_staff.min()
        metrics['max_hours_per_staff'] = hours_per_staff.max()
        
        # Weekly utilization
        weekly_hours = []
        for s in range(self.num_staff):
            staff_schedule = schedule_df[schedule_df['staff_id'] == s]
            
            for week_start in range(0, self.planning_horizon_days, 7):
                week_end = min(week_start + 7, self.planning_horizon_days)
                week_schedule = staff_schedule[(staff_schedule['day'] >= week_start) & 
                                               (staff_schedule['day'] < week_end)]
                weekly_hours.append(week_schedule['length'].sum())
        
        weekly_hours = pd.Series(weekly_hours)
        weekly_hours = weekly_hours[weekly_hours > 0]  # Only consider weeks with scheduled hours
        
        metrics['avg_weekly_hours'] = weekly_hours.mean()
        metrics['max_weekly_hours'] = weekly_hours.max()
        
        # Shift type distribution
        shift_counts = schedule_df.groupby('length').size()
        metrics['shift_distribution'] = shift_counts.to_dict()
        
        # Calculate overtime (hours > target per week)
        target_weekly_hours = self.max_weekly_hours * 0.9  # 90% of max weekly hours
        overtime_hours = weekly_hours[weekly_hours > target_weekly_hours] - target_weekly_hours
        metrics['total_overtime_hours'] = overtime_hours.sum()
        
        # Calculate the number of shifts per industry knowledge
        industry_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per company knowledge
        company_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per process knowledge
        process_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per product knowledge
        product_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per service knowledge
        service_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per market knowledge
        market_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per customer knowledge
        customer_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per competitor knowledge
        competitor_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per supplier knowledge
        supplier_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per regulatory knowledge
        regulatory_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per compliance knowledge
        compliance_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per safety knowledge
        safety_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per security knowledge
        security_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per environmental knowledge
        environmental_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per ethical knowledge
        ethical_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per cultural knowledge
        cultural_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per global knowledge
        global_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per political knowledge
        political_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per economic knowledge
        economic_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per social knowledge
        social_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per technological knowledge
        technological_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per legal knowledge
        legal_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per financial knowledge
        financial_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per accounting knowledge
        accounting_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per marketing knowledge
        marketing_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per sales knowledge
        sales_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per operations knowledge
        operations_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per human resources knowledge
        human_resources_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per information technology knowledge
        information_technology_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per research and development knowledge
        research_and_development_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per engineering knowledge
        engineering_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per manufacturing knowledge
        manufacturing_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per supply chain knowledge
        supply_chain_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per logistics knowledge
        logistics_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per procurement knowledge
        procurement_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        # Calculate the number of shifts per quality control knowledge
        quality_control_knowledge_counts = schedule_df['staff_id'].apply(lambda x: 'High' if x < 4 else 'Medium' if x < 8 else 'Low').value_counts()
        
        return metrics
    
    def visualize_schedule(self, schedule_df, days_to_show=7):
        """Visualize the schedule for a specified number of days"""
        if schedule_df is None or len(schedule_df) == 0:
            print("No schedule available to visualize.")
            return
        
        # Limit to the specified number of days
        days_to_show = min(days_to_show, self.planning_horizon_days)
        schedule_subset = schedule_df[schedule_df['day'] < days_to_show]
        
        # Create datetime objects for better visualization
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        plt.figure(figsize=(15, 10))
        
        staff_ids = sorted(schedule_subset['staff_id'].unique())
        
        for idx, staff_id in enumerate(staff_ids):
            staff_schedule = schedule_subset[schedule_subset['staff_id'] == staff_id]
            
            for _, shift in staff_schedule.iterrows():
                day = shift['day']
                start_time = base_date + timedelta(days=day, hours=shift['start_hour'])
                duration = shift['length']
                
                plt.barh(staff_id, duration, left=day * 24 + shift['start_hour'], height=0.5,
                        color=f'C{int(shift["length"]) % 10}', alpha=0.7,
                        edgecolor='black', linewidth=1)
                
                # Add shift label
                center_x = day * 24 + shift['start_hour'] + duration / 2
                plt.text(center_x, staff_id, f"{int(shift['start_hour'])}-{int(shift['end_hour'])}",
                        ha='center', va='center', fontsize=8, color='black')
        
        # Add day separators
        for day in range(days_to_show + 1):
            plt.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)
        
        # Labels for each day
        day_labels = [(base_date + timedelta(days=d)).strftime('%a\n%m/%d') for d in range(days_to_show)]
        plt.xticks([d * 24 + 12 for d in range(days_to_show)], day_labels)
        
        plt.yticks(staff_ids, [f'Staff {s}' for s in staff_ids])
        plt.xlabel('Time (hours)')
        plt.ylabel('Staff')
        plt.title('NephroPlus Staff Schedule')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return plt

# Example usage
def run_nephroplus_scheduler():
    # Initialize the scheduler with NephroPlus parameters
    scheduler = NephroPlusScheduler(
        num_beds=12,
        staff_to_bed_ratio=1/3,
        cycle_time_hours=5,
        operation_hours_per_day=20,
        planning_horizon_days=28,
        max_monthly_hours=234,
        max_weekly_hours=60,
        min_gap_between_shifts_hours=10,
        rest_days_per_week=1,
        possible_shift_lengths=[12, 10, 8, 6],
        min_start_time_gap_hours=1,
        num_staff=12
    )
    
    print(f"Total staff needed: {scheduler.total_staff}")
    print(f"Staff required per shift: {scheduler.num_staff_required}")
    
    # Build and solve the model
    scheduler.build_model()
    success = scheduler.solve()
    
    if success:
        # Get and analyze the schedule
        schedule = scheduler.get_schedule()
        
        # Print sample of the schedule
        print("\nSample Schedule (first 10 entries):")
        print(schedule.head(10))
        
        # Calculate and print metrics
        metrics = scheduler.calculate_metrics(schedule)
        print("\nSchedule Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Visualize the first week of the schedule
        plt = scheduler.visualize_schedule(schedule, days_to_show=7)
        plt.show()
        
        return schedule, metrics
    else:
        print("Failed to find a feasible schedule.")
        return None, None

if __name__ == "__main__":
    schedule, metrics = run_nephroplus_scheduler()
