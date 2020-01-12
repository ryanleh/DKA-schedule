# DKA-Schedule
A genetic algorithm for solving a constrained scheduling problem for my A Cappella group. In particular, given a when2meet for individuals split up between 4 sections, schedule them into audition slots such that:
 * Each hour has representation from all 4 sections
 * Individuals have a relatively equal number of assigned hours
 * Individuals' assigned hours in a given day are consecutive
 * Individuals don't work over a set number of hours per day

Using a genetic algorithm is a bit overkill for this problem since the search space is relatively small, but I wanted to try out [DEAP](https://github.com/deap/deap).

## Usage

Consider scheduling a week of auditions with 10 hours of slots per day. To parse the when2meet responses into a CSV, run:
```bash
python3 when2meet2csv.py $when2meet_url 7 10
```
<p align="center"><i>(DeCadence: delete the parts of the CSV representing callbacks since everyone should be there. In this example, that leaves you with 5 days instead of 7)</i></p>

Next, simply run
```bash
python3 schedule.py -d 5 -s 10 -f when2meet.csv
```
enter in the corresponding section for each individual, and an optimized schedule will be outputted as `schedule.csv`! 

Most likely, some tuning of the hyperparameters will be needed to get the best possible results. To avoid retyping section assignments, rerun the solver as follows:
```bash
python3 schedule.py -d 5 -s 10 -p schedule.parts
```
This uses the `schedule.parts` file that is automatically generated once the section assignments are inputted the first time.


