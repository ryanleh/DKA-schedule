"""
Export when2meet to csv file

Created on: January 7, 2020
    Author: ryanleh
"""
import argparse
import csv
import datetime
import re
import requests

def create_csv(url, days, hours):
    # Pull when2meet html
    r = requests.get(url)
    elements = int(max(re.findall(r'TimeOfSlot\[(\d+)', r.text), key=int)) + 1
    assert days * hours * 4 == elements, "Provided days and hours don't match when2meet"

    # Create dictionary mapping name to id
    iden_to_names = {}
    # Remove any duplicates via set
    iden_list = list(set(re.findall(r"\'(\D+)\';PeopleIDs\[\d+\]\s=\s(\d+)", r.text))) 
    for name, iden in iden_list:
        iden_to_names[iden] = name

    # Create dictionary mapping time slot to names
    schedule = {}
    for time_slot in range(elements):
        iden_list = re.findall(rf"AvailableAtSlot\[{time_slot}\].push\((\d+)", r.text)
        schedule[time_slot] = [iden_to_names[iden] for iden in iden_list]

    with open('when2meet.csv', 'w', newline='') as csvfile:
        fieldnames = ['Time'] + list(f"Day {i}" for i in range(days))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row_time in range(4*hours):
            row = {}
            row['Time'] = str(datetime.timedelta(minutes=row_time*15))
            for day in range(days):
                row[f"Day {day}"] = ";".join(schedule[row_time + day*(4*hours)])
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help='The when2meet URL', type=str)
    parser.add_argument('days', help='The number of days', type=int)
    parser.add_argument('hours', help='The number of hours per day', type=int)
    args = parser.parse_args()

    create_csv(args.url, args.days, args.hours)

