import json
import numpy as np
from typing import List, Tuple
from collections import defaultdict

# Read JSON file
with open('tiny.json') as file:
    data = json.load(file)


class Machine:
    id = 1

    def __init__(self) -> None:
        self.id = Machine.id
        Machine.id += 1
        self.map_operator_task = defaultdict(set)

    def __repr__(self) -> str:
        return f'Machine {self.id}'

    def __hash__(self) -> int:
        return hash(self.id)

    def add(self, operator, task):
        self.map_operator_task[task].add(operator)


class Job:
    def __init__(self, id, release_date, due_date, weight) -> None:
        self.id = id
        self.release_date = release_date
        self.due_date = due_date
        self.weight = weight
        self.sequence_tasks = []

    def __repr__(self) -> str:
        return f'Job {self.id}, tasks: {self.sequence_tasks}'

    def __hash__(self) -> int:
        return hash(self.id)


class Task:
    def __init__(self, id, p_t) -> None:
        self.id = id
        self.processing_time = p_t
        self.machines = set([])
        self.operators = set([])
        self.jobs = set([])

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f'Task {self.id}, processing time = {self.processing_time}, machines = {self.machines}, operators = {self.operators}'


class Operator:
    id = 1

    def __init__(self) -> None:
        self.id = Operator.id
        Operator.id += 1
        self.map_machine_task = defaultdict(set)

    def __repr__(self) -> str:
        return f'Operator {self.id}'

    def __hash__(self) -> int:
        return hash(self.id)

    def add(self, machine, task):
        self.map_machine_task[task].add(machine)


# Extract parameters and data from JSON
parameters = data["parameters"]
size = data["parameters"]["size"]
nb_machines = size["nb_machines"]
nb_operators = size["nb_operators"]
jobs = data["jobs"]
tasks = data["tasks"]
alpha = parameters["costs"]["unit_penalty"]
beta = parameters["costs"]["tardiness"]

machines = [Machine() for _ in range(nb_machines)]
operators = [Operator() for _ in range(nb_operators)]


def link_tasks(this_task: Task, machines_list) -> Task:
    for mach in machines_list:
        m = machines[mach['machine'] - 1]
        for i in mach['operators']:
            m.add(operators[i - 1], this_task)
            operators[i - 1].add(m, this_task)
            this_task.operators.add(operators[i - 1])
        this_task.machines.add(m)
    return this_task


tasks_list = []
for task in tasks:
    this_task = Task(task['task'], task['processing_time'])
    tasks_list.append(link_tasks(this_task, task['machines']))


def link_jobs(job: Job, sequence) -> Job:
    for task_id in sequence:
        t = tasks_list[task_id - 1]
        job.sequence_tasks.append(tasks_list[task_id - 1])
        t.jobs.add(job)
    return job


jobs_list = []
for job in jobs:
    j = Job(job['job'], job['release_date'], job['due_date'], job['weight'])
    jobs_list.append(link_jobs(j, job['sequence']))


def check_solution(solution: List[Tuple[Job, Task, Machine, Operator]]):
    for job, task, machine, operator in solution:
        if task not in job.sequence_tasks:
            return False
        if machine not in task.machines:
            return False
        if operator not in machine.map_operator_task[task]:
            return False
    return True


def cost(solution: List[Tuple[Job, Task, Machine, Operator]]):
    cost = 0
    time_per_machine = {m: 0 for m in machines}
    time_per_operator = {o: 0 for o in operators}
    time_per_job = {j: j.release_date for j in jobs_list}
    for job, task, machine, operator in solution:
        start_time = max(time_per_machine[machine], time_per_operator[operator], time_per_job[job])
        time_per_job[job] += max(start_time - time_per_job[job], 0) + task.processing_time
        time_per_machine[machine] += max(start_time - time_per_machine[machine], 0) + task.processing_time
        time_per_operator[operator] += max(start_time - time_per_operator[operator], 0) + task.processing_time
        if task == job.sequence_tasks[-1]:  # If it is over
            completion_time = time_per_job[job] + job.release_date
            tardiness_time = max(completion_time - job.due_date, 0)
            unit_penalty_cost = 1 if completion_time > job.due_date else 0
            cost += job.weight * (completion_time + alpha * unit_penalty_cost + beta * tardiness_time)
    return cost


def get_initial_solution():
    s = []
    for j in jobs_list:
        for t in j.sequence_tasks:
            m = np.random.choice(list(t.machines))
            o = np.random.choice(list(m.map_operator_task[t]))
            s.append((j, t, m, o))
    return s


def pertubate_solution(solution):
    pertub_sol = solution.copy()
    random_task_in_solution = np.random.choice(len(solution))
    job, task, _, _ = solution[random_task_in_solution]
    new_m = np.random.choice(list(task.machines))
    new_o = np.random.choice(list(new_m.map_operator_task[task]))
    pertub_sol[random_task_in_solution] = (job, task, new_m, new_o)
    return pertub_sol


def simulated_annealing(initial_solution, max_iterations, initial_temp, cooling_rate):
    current_solution = initial_solution
    current_cost = cost(current_solution)
    temp = initial_temp
    #    print(current_solution)
    for i in range(max_iterations):
        neighbor_solution = pertubate_solution(current_solution)
        if not check_solution(neighbor_solution):
            raise ValueError('Not a solution')
        neighbor_cost = cost(neighbor_solution)
        delta = neighbor_cost - current_cost
        if delta < 0:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
        else:
            prob = np.exp(-delta / temp)
            if np.random.random() < prob:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
        temp *= cooling_rate
    return current_solution, current_cost


def get_solution_json(solution: List[Tuple[Job, Task, Machine, Operator]]):
    solution_json = []
    time_per_machine = {m: 0 for m in machines}
    time_per_operator = {o: 0 for o in operators}
    time_per_job = {j: j.release_date for j in jobs_list}
    for job, task, machine, operator in solution:
        start_time = max(time_per_machine[machine], time_per_operator[operator], time_per_job[job])
        time_per_job[job] += max(start_time - time_per_job[job], 0) + task.processing_time
        time_per_machine[machine] += max(start_time - time_per_machine[machine], 0) + task.processing_time
        time_per_operator[operator] += max(start_time - time_per_operator[operator], 0) + task.processing_time
        solution_json.append({"job": job.id,
                              "task": task.id,
                              "start": start_time,
                              "operator": operator.id,
                              "machine": machine.id})

    return solution_json


# Define the parameters for the simulated annealing algorithm
max_iterations = 10000
initial_temp = 100
cooling_rate = 0.95
# Run  the simulated annealing algorithm
best_solution, best_cost = simulated_annealing(get_initial_solution(), max_iterations, initial_temp, cooling_rate)


print("Cost:", best_cost)

solution_json = get_solution_json(best_solution)
print(solution_json)

with open("solution_tiny.json", "w", encoding="UTF-8") as outfile:
    outfile.write(json.dumps(solution_json, indent=4))