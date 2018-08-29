import main


# python main_test.py
def test_index():
    main.app.testing = True
    client = main.app.test_client()

    r = client.get('/')
    assert r.status_code == 200
    assert 'Welcome' in r.data.decode('utf-8')


if __name__ == "__main__":
    test_index()
