import csv
import json

def convert_csv_to_training_format(input_csv, output_file):
    system_message = {
        "role": "system",
        "content": "You are a call center AI. A user has contacted you for help. "
                   "You need to read the user's message and provide a response. "
    }

    with open(input_csv, 'r', encoding='utf-8') as csvfile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for row in reader:
            medical_report = row[0]
            extracted_json = row[1]

            training_example = {
                "messages": [
                    system_message,
                    {"role": "user", "content": medical_report},
                    {"role": "assistant", "content": extracted_json}
                ]
            }
            outfile.write(json.dumps(training_example) + '\n')


# Prepare training data
convert_csv_to_training_format("./data/data.csv", "./data/training_data.jsonl")
