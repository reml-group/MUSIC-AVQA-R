import json
import yaml

def count_test_set(config):
    test_path = config['path']['test_data']['extend-tail']
    with open(file=test_path, mode='r') as f:
        test_set = json.load(f)

    count = {}
    for que in test_set:
        if count.get(que['type'], None) is None:
            count[que['type']] = 1
        else:
            count[que['type']] += 1

    print(count)

if __name__ == "__main__":
   with open(file='/home/majie/code/ravqa/option.yaml', mode='r') as f:
       config = yaml.safe_load(f)
   count_test_set(config)