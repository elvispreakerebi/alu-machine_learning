#!/usr/bin/env python3
"""
Return list of schools having a specific topic.
"""
def schools_by_topic(mongo_collection, topic):
    """
    Return list of school documents that have topic in their topics array.
    topic: string to search for
    """
    return list(mongo_collection.find({"topics": topic}))
