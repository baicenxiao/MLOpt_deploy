from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

pos_examples = [{'age': 55,
                 'workclass': '?',
                 'fnlgt': 141807,
                 'education': 'HS-grad',
                 'education-num': 9,
                 'marital-status': 'Never-married',
                 'occupation': '?',
                 'relationship': 'Not-in-family',
                 'race': 'White',
                 'sex': 'Male',
                 'capital-gain': 13550,
                 'capital-loss': 0,
                 'hours-per-week': 40,
                 'native-country': 'United-States',
                 'salary': '>50K'}
                ]

neg_examples = [{'age': 21,
                 'workclass': 'Private',
                 'fnlgt': 216181,
                 'education': 'Some-college',
                 'education-num': 10,
                 'marital-status': 'Never-married',
                 'occupation': 'Sales',
                 'relationship': 'Own-child',
                 'race': 'White',
                 'sex': 'Male',
                 'capital-gain': 0,
                 'capital-loss': 0,
                 'hours-per-week': 35,
                 'native-country': 'United-States',
                 'salary': '<=50K'}
                ]


def test_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Welcome to the salary predictor"}


def test_pos_outcome():
    for neg_example in neg_examples:
        response = client.post("/score", json=neg_example)
        assert response.status_code == 200
        assert response.json()['prediction'] == neg_example['salary']


def test_neg_outcome():
    for pos_example in pos_examples:
        response = client.post("/score", json=pos_example)
        assert response.status_code == 200
        assert response.json()['prediction'] == pos_example['salary']
