from pathlib import Path

env_config = {'instance_path': str(Path(__file__).parent.absolute()) + '/input'}
instance_path = env_config['instance_path']

instance_file = open(instance_path, 'r')
line_str = instance_file.readline()
line = 1 

while line_str:
    split_data = line_str.split()
    i = 0
    temp_str = ''
    while i < (len(split_data) - 1):
        temp_str += '(' + split_data[i] + ',' + split_data[i+1] + ')'
        i += 2 
        if i <= len(split_data)-2 :
            temp_str += ','
    temp_str = '[' + temp_str +']' + ','
    print(temp_str)
    line_str = instance_file.readline()