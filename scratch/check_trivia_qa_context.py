from datasets import load_dataset

ds = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
for ex in ds:
    print("entity_pages:", ex["entity_pages"])
    # Let's print the keys inside entity_pages if they exist
    if ex["entity_pages"]:
        print("keys inside entity_pages:", ex["entity_pages"].keys())
        print("wiki_context:", ex["entity_pages"]["wiki_context"][:1] if "wiki_context" in ex["entity_pages"] else "Not found")
    print("search_results:", ex["search_results"])
    break
