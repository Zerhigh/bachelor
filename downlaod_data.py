import requests

with open('hurricane_ian.txt', 'r') as file:
    lines = file.readlines()

lines = [line.strip().replace('\n', '') for line in lines[30:150]]

for i, line in enumerate(lines):
    URL = line
    title = URL.split('/')[-1]
    response = requests.get(URL)
    save_path = 'C:/Users/Samuel/Desktop/TU/BachelorArbeit/automatic_download_maxar_ian/'
    open(f"{save_path}_{30+i}_{title}", "wb").write(response.content)
    print(f'saved _{30+i}_{title}')