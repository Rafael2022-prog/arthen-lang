import json
from parser.ai_native_parser import AIOptimizedParser, AIToken

def load_source(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def find_aitoken(obj, path='root') -> bool:
    if isinstance(obj, AIToken):
        print('Found AIToken at', path)
        return True
    if isinstance(obj, dict):
        for k, v in obj.items():
            if find_aitoken(v, f'{path}.{k}'):
                return True
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if find_aitoken(v, f'{path}[{i}]'):
                return True
    return False

if __name__ == '__main__':
    src = load_source('examples/defi_liquidity_pool.arthen')
    parser = AIOptimizedParser()
    result = parser.parse(src)
    print('Top-level keys:', list(result.keys()))
    print('Scanning for AIToken instances...')
    found = find_aitoken(result)
    print('Found?', found)
    if not found:
        # Attempt json dump
        s = json.dumps(result)
        print('JSON length:', len(s))