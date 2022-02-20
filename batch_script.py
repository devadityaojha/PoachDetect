import json
import pandas as pd


file_list = ["./tasks.json", "./tasks1.json", "./tasks2.json"]
tasks = json.load(open("./tasks.json"))

data = []

for file_name in file_list:
    with open(file_name) as file:
        tasks = json.load(file)
        for i, task in enumerate(tasks):
            try:
                data.append([task['task_id'], task['params']['attachment'], task['metadata']['filename'], int(task['response']['annotations']['gunshot'][0] == 'yes')])
            except:
                continue

df = pd.DataFrame(data, columns=['task_id', 'attachment', 'filename', 'gunshot'])

print(df)
df.to_csv("./audio_labels.csv")
# print(tasks)