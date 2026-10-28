from tools.search_agricultural_knowledge import search_agricultural_knowledge

def test_empty_query_returns_empty_list():
    assert search_agricultural_knowledge("   ") == []
