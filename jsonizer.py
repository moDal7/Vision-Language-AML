import json

input_path= "data/Labeled_PACS/groupe6AML.txt"
output_path= "out1.json"


data= ""
with open(input_path, "r") as fp:
    for line in fp:
        for idx, c in enumerate(line):
            if c=="'": 
                if line[idx-1].isalpha() and line[idx+1].isalpha():
                    data+= "'"
                else:
                    data+= '"'
            else:
                data+= c

with open(output_path, "w") as fp:
    fp.writelines(data)

data= ""
with open(output_path, "r") as fp:
    data= json.load(fp)

js= json.dumps(data, indent= 2)
with open(output_path, "w") as fp:
    fp.write(js)

data= ""
with open(output_path, "r") as fp:
    data= json.load(fp)

print(data[20])
