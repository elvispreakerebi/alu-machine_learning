#!/usr/bin/env python3
"""
Insert a new document in a MongoDB collection from kwargs.
Returns the new _id.
"""
def insert_school(mongo_collection, **kwargs):
    """
    Insert a new document in mongo_collection with kwargs as attributes.
    Returns the new document's _id.
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
