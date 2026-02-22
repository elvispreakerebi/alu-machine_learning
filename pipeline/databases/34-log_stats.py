#!/usr/bin/env python3
"""
Provide stats about Nginx logs stored in MongoDB (logs.nginx).
"""
from pymongo import MongoClient


def main():
    """
    Display log stats: total count, method counts, status check count.
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx = client.logs.nginx

    total = nginx.count_documents({})
    print("{} logs".format(total))

    print("Methods:")
    for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        count = nginx.count_documents({"method": method})
        print("    method {}: {}".format(method, count))

    status = nginx.count_documents({"method": "GET", "path": "/status"})
    print("{} status check".format(status))


if __name__ == "__main__":
    main()
