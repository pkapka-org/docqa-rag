import gc

from docqa.chunking import chunk_documents
from docqa.loaders import load_bytes
from docqa.store import DocumentStore


def chunks(name, text):
    return chunk_documents(load_bytes(name, text.encode()), 200, 20)


def test_search_ranks_best_match_first(store):
    store.add(chunks("a.txt", "cats purr loudly"))
    store.add(chunks("b.txt", "dogs bark loudly"))
    hits = store.search("dogs bark", 2)
    assert hits[0][0].metadata["source"] == "b.txt"
    assert hits[0][1] > hits[1][1]


def test_same_file_twice_is_indexed_once(store):
    assert store.add(chunks("a.txt", "hello there world")) == 1
    assert store.add(chunks("a.txt", "hello there world")) == 0
    assert store.sources() == {"a.txt": 1}


def test_new_version_replaces_old(store):
    store.add(chunks("a.txt", "the old price is ten"))
    store.add(chunks("a.txt", "the new price is twenty"))
    assert store.sources() == {"a.txt": 1}
    assert "twenty" in store.search("price", 1)[0][0].page_content


def test_two_files_with_the_same_content_both_stay(store):
    store.add(chunks("a.txt", "identical text here"))
    store.add(chunks("b.txt", "identical text here"))
    assert store.sources() == {"a.txt": 1, "b.txt": 1}
    store.remove("a.txt")
    assert store.sources() == {"b.txt": 1}


def test_remove_and_clear(store):
    store.add(chunks("a.txt", "one two three"))
    store.remove("missing.txt")
    store.clear()
    assert store.sources() == {}
    assert store.search("one", 3) == []


def test_in_memory_stores_are_isolated(embeddings):
    first, second = DocumentStore(embeddings), DocumentStore(embeddings)
    first.add(chunks("private.txt", "my secret salary"))
    assert second.sources() == {}
    assert second.search("secret salary", 3) == []


def test_in_memory_collection_is_freed(embeddings):
    store = DocumentStore(embeddings)
    client, name = store._db._client, store._db._collection.name
    assert name in [c.name for c in client.list_collections()]
    del store
    gc.collect()
    assert name not in [c.name for c in client.list_collections()]


def test_persistent_store_survives_reopening(embeddings, tmp_path):
    DocumentStore(embeddings, path=str(tmp_path)).add(chunks("a.txt", "kept on disk"))
    assert DocumentStore(embeddings, path=str(tmp_path)).sources() == {"a.txt": 1}
