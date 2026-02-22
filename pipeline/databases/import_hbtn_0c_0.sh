#!/bin/bash
# Import hbtn_0c_0 dump into MySQL. Place hbtn_0c_0.sql in this directory first.
# Download from your project's "download" link, then run: ./import_hbtn_0c_0.sh

set -e
cd "$(dirname "$0")"
DUMP="hbtn_0c_0.sql"

if [ ! -f "$DUMP" ]; then
  echo "Error: $DUMP not found. Download it from your project and place it here."
  exit 1
fi

echo "CREATE DATABASE IF NOT EXISTS hbtn_0c_0;" | mysql -h127.0.0.1 -P3307 -uroot -proot 2>/dev/null
mysql -h127.0.0.1 -P3307 -uroot -proot hbtn_0c_0 < "$DUMP" 2>/dev/null
echo "Imported $DUMP into hbtn_0c_0."
