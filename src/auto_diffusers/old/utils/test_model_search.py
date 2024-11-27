from ..model_path.perform_path_search import Search_cls


test_cases = [
    {'search_word': 'any', 'auto': True, 'model_format': 'all'},
    {'search_word': 'any', 'auto': True, 'model_format': 'diffusers'},
    {'search_word': 'any', 'auto': True, 'model_format': 'single_file'},
    {'search_word': 'any', 'auto': False, 'model_format': 'all'},
    {'search_word': 'any', 'auto': False, 'model_format': 'diffusers'},
    {'search_word': 'any', 'auto': False, 'model_format': 'single_file'}
]
test_model_search = Search_cls()
for case in test_cases:
    search_word = case['search_word']
    auto = case['auto']
    model_format = case['model_format']
    print(f'Running model_search with search_word={search_word}, auto={auto}, model_format={model_format}')
    test_model_search(search_word=search_word, auto=auto, model_format=model_format)
