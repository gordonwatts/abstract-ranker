from abstract_ranker.utils import generate_ranking_csv_filename


def test_filename():
    event = {"startDate": {"date": "2024-06-24"}, "title": "ACAT 2024"}

    assert generate_ranking_csv_filename(event) == "2024-06-24 - ACAT 2024.csv"
