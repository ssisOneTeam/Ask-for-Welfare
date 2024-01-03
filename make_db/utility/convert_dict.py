import json

def process_string(s: str):
    s = s.strip()
    s = s.replace('\n', '')
    s = s.rstrip(',')
    return s

def key_process(s: str):
	s = s.strip()
	s = s.replace('\n', '')
	s = s.replace(' ', '')
	return s

def remove_duplicates_from_values(dictionary):
    for key, value in dictionary.items():
        unique_values = list(set(value.split(',')))
        dictionary[key] = ','.join(unique_values)

if __name__ == "__main__":
	meta = {}
	file_path = './result.txt'
	with open(file_path, 'r') as file:
		contents = file.readlines()

	for item in contents:
		temp = item.split(':')
		k, v = key_process(temp[0]), process_string(temp[1])
		if k in meta:
			meta[k] += f", {v}"
		else:
			meta[k] = v

	remove_duplicates_from_values(meta)
	print(len(meta))
	print(meta)

	with open('./metadata.json', 'w') as f:
		json.dump(meta, f, ensure_ascii=False, indent=4)