from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
import time

if __name__ == '__main__':
    milvus_uri = 'http://10.15.8.71:19530'

    connections.connect("default", uri=milvus_uri)
    print(f"Connecting to DB: {milvus_uri}")
    print("\nlist collections:")
    if True:
        collections = utility.list_collections()
        print(collections)
        for col in collections:
            collection = Collection(col)
            print(f"{col} has {collection.num_entities} entities（{collection.num_entities/1000000}M）, indexed {collection.has_index()}")

    TEST = "VectorDBBenchCollection"
    collection = Collection(TEST)
    utility.wait_for_index_building_complete(TEST)
    print("created index")
    try:
        t = int(time.time())
        collection.load()
        print(f"loaded index into memory after {int(time.time()) - t} seconds")
    except Exception as e:
        print("load into memory failed")
        print(e)
    # rename_result = utility.rename_collection("VectorDBBenchCollection", "VectorDBBenchCollection10M")
    # print(f"Rename VectorDBBenchCollection to VectorDBBenchCollection10M {rename_result}")
    # utility.rename_collection("VectorDBBenchCollection", "VectorDBBenchCollection1M")
    # print(utility.list_collections())
    # utility.drop_collection("VectorDBBenchCollection10MFrom220Branch")

    # utility.drop_collection("VectorDBBenchCollection")
    # print(utility.list_collections())