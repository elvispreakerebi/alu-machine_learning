#!/usr/bin/env python3
"""
List all documents in a MongoDB collection.
Returns empty list if no documents.
"""
def list_all(mongo_collection):
    """
    List all documents in mongo_collection.
    Return empty list if no document in the collection.
    """
    return list(mongo_collection.find())
