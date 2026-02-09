import json
from collections import defaultdict

def analyze_context_json(input_file, output_file):
    # Track data per lecture
    datasets = defaultdict(lambda: {
        'pages': set(),
        'chunks': 0,
        'source': None
    })
    
    # Read the JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                
                lecture_id = chunk['lecture_id']
                
                # Add pages for THIS lecture
                datasets[lecture_id]['pages'].update(chunk['page_numbers'])
                
                # Track source
                if datasets[lecture_id]['source'] is None:
                    datasets[lecture_id]['source'] = chunk['source_pdf']
                
                # Count chunks
                datasets[lecture_id]['chunks'] += 1
    
    # Calculate totals
    total_pages = sum(len(data['pages']) for data in datasets.values())
    total_chunks = sum(data['chunks'] for data in datasets.values())
    sources = [data['source'] for data in datasets.values()]
    
    # Prepare output
    output_data = {
        'content_format': 'JSONL',
        'total_pages': total_pages,
        'total_chunks': total_chunks,
        'sources': sources
    }
    
    # Write output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Total pages (sum across all lectures): {total_pages}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total lectures/sources: {len(sources)}")

if __name__ == "__main__":
    analyze_context_json('/home/alrabah2/graphika/graphika/algo_output_llama3b/chunks.jsonl', 'output.json')