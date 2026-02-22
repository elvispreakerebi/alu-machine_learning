#!/usr/bin/env python3
"""
Update the topics of school documents by name.
"""
def update_topics(mongo_collection, name, topics):
    """
    Change all topics of school documents matching name.
    name: school name to update
    topics: list of topic strings
    """
    mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}})
