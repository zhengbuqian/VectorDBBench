from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema


def load_collection(v):
    collection = Collection(v)
    print(collection.num_entities)
    print(collection.has_index())
    try:
        collection.load()
        print("loaded")
    except Exception as e:
        print("load into memory failed")
        print(e)

def release_collection(v):
    collection = Collection(v)
    print(collection.num_entities)
    print(collection.has_index())
    try:
        collection.release()
        print("released")
    except Exception as e:
        print("release from memory failed")
        print(e)

def rename(src, dst):
    print(f"Renaming {src} to {dst}")
    rename_result = utility.rename_collection(src, dst)
    print(f"Renamed {src} to {dst}: {rename_result}")


if __name__ == '__main__':
    milvus_uri = 'http://10.15.8.71:19530'

    connections.connect("default",
                        uri=milvus_uri)
    print(f"Connecting to DB: {milvus_uri}")
    print("\nlist collections:")
    collections = utility.list_collections()
    print(collections)
    V10M = 'VectorDBBenchCollection10M'
    V1M = 'VectorDBBenchCollection1M'
    V = 'VectorDBBenchCollection'
    collections.remove(V)
    target = collections[0]
    other = ''
    if target == V1M:
        other = V10M
    else:
        other = V1M
    print(f"switching to {target} from {other}")
    release_collection(V)

    rename(V, other)
    rename(target, V)

    collections = utility.list_collections()
    print(f"Collections are now: {collections}")

    load_collection(V)
