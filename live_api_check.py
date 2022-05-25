from test_main import pos_examples, neg_examples
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", default='https://mlopt-deploy.herokuapp.com/score')

args = parser.parse_args()
endpoint = args.endpoint

print(f'testing using endpoint: {endpoint}')

def check_api(example_data):
    response = requests.post(endpoint, json=example_data)
    msg = (f'api result: {response.status_code} '
           f'prediction: {response.json()} '
           f'expected: {example_data["salary"]}')
    print(msg)


if __name__ == "__main__":
    for example in pos_examples + neg_examples:
        check_api(example)