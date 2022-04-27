"""Minimal jobshop example."""
import collections
from ortools.sat.python import cp_model


def main():
    """Minimal jobshop problem."""
    # Data.
    # jobs_data = [  # task = (machine_id, processing_time).
    #     [(0, 3), (1, 2), (2, 2)],  # Job0
    #     [(0, 2), (2, 1), (1, 4)],  # Job1
    #     [(1, 4), (2, 3)]  # Job2
    # ]
    jobs_data = [ 
            [(9,86),(14,60),(4,10),(13,59),(10,65),(3,94),(7,71),(8,25),(0,98),(5,49),(1,43),(2,8),(12,90),(6,21),(11,73)],
            [(10,68),(8,28),(11,38),(14,36),(3,93),(13,35),(9,37),(7,28),(4,62),(2,86),(6,65),(1,11),(5,20),(12,82),(0,23)],
            [(7,33),(0,67),(6,96),(5,91),(14,83),(13,81),(2,60),(11,88),(4,20),(12,62),(1,22),(9,79),(3,38),(10,40),(8,82)],
            [(9,13),(11,14),(14,73),(0,88),(1,24),(8,16),(5,78),(10,70),(12,53),(4,68),(13,73),(3,90),(6,58),(7,7),(2,4)],
            [(11,93),(4,52),(13,63),(3,13),(8,19),(1,41),(10,71),(12,59),(2,19),(14,60),(6,85),(7,99),(0,73),(9,95),(5,19)],
            [(5,62),(2,60),(1,93),(10,16),(0,10),(4,72),(8,88),(14,69),(6,58),(3,41),(9,46),(7,63),(11,76),(12,83),(13,62)],
            [(5,50),(10,68),(13,90),(0,34),(9,44),(8,5),(1,8),(11,25),(14,70),(7,53),(12,78),(2,92),(6,62),(4,85),(3,70)],
            [(12,60),(0,64),(9,92),(3,44),(13,63),(6,91),(5,21),(7,1),(2,96),(14,19),(11,59),(8,12),(10,41),(1,11),(4,94)],
            [(11,93),(10,46),(5,51),(13,37),(1,91),(9,90),(8,63),(7,40),(3,68),(6,13),(0,16),(2,83),(14,49),(12,24),(4,23)],
            [(2,5),(14,35),(3,21),(10,14),(6,66),(1,3),(0,6),(13,98),(11,63),(4,64),(5,76),(8,94),(7,17),(12,62),(9,37)],
            [(11,35),(14,42),(13,62),(5,68),(4,73),(9,27),(1,52),(6,39),(12,41),(0,25),(2,9),(8,34),(10,50),(3,41),(7,98)],
            [(12,23),(3,32),(10,35),(8,10),(4,29),(7,68),(13,20),(11,8),(14,58),(1,62),(2,39),(0,32),(5,8),(6,33),(9,91)],
            [(8,28),(13,31),(5,3),(0,28),(11,66),(9,59),(4,24),(12,45),(1,81),(10,8),(6,44),(2,42),(7,2),(14,23),(3,53)],
            [(2,11),(5,93),(4,27),(3,59),(9,62),(1,23),(11,23),(13,7),(7,77),(6,64),(10,60),(14,97),(0,36),(8,53),(12,72)],
            [(1,36),(10,98),(4,38),(2,24),(0,84),(7,47),(6,72),(9,1),(11,91),(12,85),(5,68),(14,42),(3,20),(13,30),(8,30)]

    ]
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution:')
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                                       job=job_id,
                                       index=task_id,
                                       duration=task[1]))

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_task_%i' % (assigned_task.job,
                                           assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
        print(output)
    else:
        print('No solution found.')

    # Statistics.
    print('\nStatistics')
    print('  - conflicts: %i' % solver.NumConflicts())
    print('  - branches : %i' % solver.NumBranches())
    print('  - wall time: %f s' % solver.WallTime())


if __name__ == '__main__':
    main()