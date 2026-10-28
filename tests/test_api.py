def test_api_module_imports():
    from api.main import app
    assert app.title == "AI Paddy Farming Intelligence API"
