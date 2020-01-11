"""
DeCadence A Cappella Audition Scheduler

Created on: January 7, 2020
    Author: ryanleh
"""
import argparse
import csv
import datetime
import itertools
import math
import numpy as np
import pickle
import random

from deap import creator, base, tools, algorithms


def gen_schedule(csv_path):
    """Generate dictionary of constraints from when2meet csv"""
    schedule = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Skip the first line
        iter(csv_reader).__next__()
        for slot, row in enumerate(csv_reader):
            for day, assigned in enumerate(row[1:]):
                names = assigned.split(";")
                # When2Meet assigns in 15 minute intervals
                hour = int(math.floor(slot / 4.0))
                for name in names:
                    if name in schedule:
                        schedule[name][day][hour] += 1
                    else:
                        schedule[name] = [[0 for _ in HOURS] for _ in DAYS]
                        schedule[name][day][hour] = 1
    # If someone can work 30 minutes or more in an hour schedule then schedule
    # them for the whole slot
    for name in schedule:
        schedule[name] = [[1 if time >= 2 else 0
                           for time in schedule[name][day]] for day in DAYS]
    return schedule

def organize_parts(schedule):
    """Separate individuals by part"""
    names = list(schedule.keys())
    sections = {
        'Bass':  {},
        'Tenor': {},
        'Alto':  {},
        'Sop':   {}
    }
    print("For each member, please enter the correct section for each member")
    section_mapping = " ".join(section for section in sections)
    for name in names:
        print()
        while True:
            print(f"{name}? ({section_mapping})")
            assignment = input()
            try:
                sections[assignment][name] = schedule[name]
                break
            except KeyError:
                print(f"Invalid assignment: {assignment}. Try again\n")
    # Write the organized schedule to a file to avoid having to type this in on
    # every run
    with open('schedule.parts', 'wb') as f:
        f.write(pickle.dumps(sections))
    return sections
        
def gen_constraints(schedule):
    """Generate constraints from a given schedule"""
    constraints = [list(list() for _ in HOURS) for _ in DAYS]
    for name in schedule:
        for day in DAYS:
            for hour in HOURS:
                if schedule[name][day][hour]:
                    constraints[day][hour].append(name) 
    return constraints

def init_individual(parent):
    """Generate a new individual with a random schedule from given constraints"""
    schedule = [list() for _ in DAYS]
    for day in DAYS:
        for slot in HOURS:
            try:
                schedule[day].append(random.choice(parent.constraints[day][slot]))
            except IndexError:
                schedule[day].append("")
    return parent(schedule)

def evaluate(schedule, individual):
    """ For a given individual, calculate values of following features:
        * Variance of allocated hours across individuals
        * Alignment of slots in a given day ie. if there are gaps in schedule
        * How many hours are worked overtime in a given day
    """
    variance = 0.0
    alignment = [1.0 for _ in DAYS]
    overworked = 0.0

    # Find the variance of normalized distribution of allocated hours
    allocated_hours = np.array([])
    for i, name in enumerate(schedule):
        allocated_hours = np.append(allocated_hours, 0)
        for assignment in itertools.chain.from_iterable(individual):
            if name == assignment:
                allocated_hours[i] += 1
    norm_hours = allocated_hours / np.linalg.norm(allocated_hours)
    variance = np.var(norm_hours)

    # Calculate alignment for each day (a percentage of how many of the hours
    # are aligned). Take the average of all the days as the final alignment
    for i, day in enumerate(individual):
        for name in schedule: 
            assigned = sum([1 for slot in day if slot == name])
            # Calculate overworked variable as an integer if a name is assigned
            # for more than OVERTIME hours in a given day
            overworked += max(assigned-OVERTIME, 0)
            if assigned != 0:
                # Keep track of how many aligned blocks thus far and if
                # currently on a block
                aligned = [0, False]
                for slot in day:
                    if slot == name and not aligned[1]:
                        aligned = [aligned[0] + 1, True]
                    if slot != name and aligned[1]:
                        aligned[1] = False
                alignment[i] *= (assigned - aligned[0] + 1) / assigned;
    return variance, np.mean(alignment), overworked

def mutate(ind, prob):
    """Mutate each scheduled day with probability prob"""
    for day in DAYS:
        if random.random() < prob:
            # Select random slot, and randomly mutate to a different valid name
            # if it exists
            hour = random.randint(0, len(HOURS)-1)
            try:
                ind[day][hour] = random.choice(
                        [name for name in ind.constraints[day][hour]
                            if name != ind[day][hour]])
            except IndexError:
                pass
    return ind,

def crossover(ind1, ind2):
    """Swap a random day assignment for two individuals"""
    day = random.randint(0, len(DAYS)-1)
    ind1[day], ind2[day] = ind2[day], ind1[day]
    return ind1, ind2

def selBest(ind, k):
    """Returns the k best-performing individuals by fitness"""
    return sorted(ind, key=lambda x: sum(x.fitness.wvalues), reverse=True)[:k]

def runtime(schedule):
    # Initialize deap
    creator.create("FitnessMax", base.Fitness, weights=(VAR_PENALTY,
                                                        ALIGN_BONUS,
                                                        OVERTIME_PENALTY))
    creator.create("Individual", list, fitness=creator.FitnessMax, constraints=gen_constraints(schedule))

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, schedule)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate, prob=0.5)

    # Generate new population and calculate fitness
    population = toolbox.population(n=POP_SIZE)
    fits = toolbox.map(toolbox.evaluate, population)
    for fit, ind in zip(fits, population):
        ind.fitness.values = fit

    top = None
    for gen in range(NGEN):
        # Create offspring population by via mating and mutation
        offspring = algorithms.varAnd(population, toolbox, cxpb=CX_PR, mutpb=MUT_PR)
        # Calculate fitness of offspring 
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        # Select the top 10% from the previous population
        carry_over = max(1, int(POP_SIZE*.1))
        pop_top10 = selBest(population, k=carry_over)
        # Select the rest to be the top from offspring
        off_top90 = selBest(offspring, k=(POP_SIZE-carry_over))
        population = pop_top10 + off_top90
        # Save the overal top individual
        top = selBest(population, k=1)[0]
        # Debugging output
        if gen % 10 == 0 or gen == NGEN-1:
            print(f"TOP SCHEDULE, GENERATION {gen}:")
            print(f"{top.fitness.values} = {sum(top.fitness.wvalues)}:")
            print(f"{top}\n")
            print("---------------")
    # Return the final schedule as an np.array
    return np.array(top)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    req = parser.add_argument_group('required arguments')
    req.add_argument('-d', '--days', required=True, type=int,
                        help='The number of days to schedule')
    req.add_argument('-s', '--slots', required=True, type=int,
                        help='The number of slots (hours) in a day to schedule')
    loader = req.add_mutually_exclusive_group(required=True)
    loader.add_argument('-w', '--when2meet', type=str,
                        help='CSV file to read when2meet schedule')
    loader.add_argument('-p', '--part_file', type=str,
                        help='Pickled dictionary of schedule organized by parts')
    
    opt = parser.add_argument_group('optional arguments')
    opt.add_argument('--size', required=False, default=1000, type=int,
                        help='Population size (default 1000)')
    opt.add_argument('--generations', required=False, default=100, type=int,
                        help='Number of generation (default 100)')
    opt.add_argument('--cx_prob', required=False, default=0.4, type=float,
                        help='Crossbreeding probability (default 0.4)')
    opt.add_argument('--mut_prob', required=False, default=0.2, type=float,
                        help='Mutation probability (default 0.2)')

    opt.add_argument('--overtime', required=False, default=4, type=int,
                        help='Number of hours before going overtime (default 4)')
    opt.add_argument('--var_penalty', required=False, default=-50.0, type=float,
                        help='Variance penalty (default -50.0)')
    opt.add_argument('--align_bonus', required=False, default=5, type=float,
                        help='Bonus for aligning hours (default 5)')
    opt.add_argument('--overtime_penalty', required=False, default=-1,
                        type=float, help='Overtime penalty (default -1)')
    args = parser.parse_args()

    # Set global constants
    global DAYS, HOURS, POP_SIZE, NGEN, CX_PR, MUT_PR, OVERTIME, VAR_PENALTY, \
           ALIGN_BONUS, OVERTIME_PENALTY
    DAYS = range(args.days)
    HOURS = range(args.slots)
    POP_SIZE = args.size
    NGEN = args.generations
    CX_PR = args.cx_prob
    MUT_PR = args.mut_prob
    OVERTIME = args.overtime
    VAR_PENALTY = args.var_penalty
    ALIGN_BONUS = args.align_bonus
    OVERTIME_PENALTY = args.overtime_penalty
  
    # Use already generated parts schedule if provided
    if args.part_file:
        with open(args.part_file, 'rb') as f:
            section_schedules = pickle.loads(f.read())
    else:
        section_schedules = organize_parts(gen_schedule(args.when2meet))

    # Create a DAYS x HOURS x SECTIONS matrix to hold the assignments
    optimal_schedule = np.array([[
        np.empty((4,), dtype=np.dtype('U30')) for _ in HOURS] for _ in DAYS])

    # Optimize the schedule for each individual section
    for i, section in enumerate(section_schedules):
        optimal_schedule[:, :, i] = runtime(section_schedules[section])
    
    # Write the resulting schedule to a csv
    with open('schedule.csv', 'w', newline='') as csvfile:
        fieldnames = ['Time'] + list(f"Day {i}" for i in DAYS)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for hour in HOURS:
            row = {}
            row['Time'] = str(datetime.timedelta(minutes=hour*60))
            for day in DAYS:
                row[f"Day {day}"] = " ".join(optimal_schedule[day, hour, :])
            writer.writerow(row)
    
    # Print out statistics for the generated schedule
    print(optimal_schedule)
    names = list(np.unique(optimal_schedule))
    for i, name in enumerate(names):
        names[i] = (name, np.char.count(optimal_schedule, name).sum())
    print("\nHour assignments: ")
    for name in names:
        print(f"{name[0]} = {name[1]}")
