# 清理所有数据库的外键孤儿数据并再次检查一致性
# 用于解决在人物扫描时报错

import sqlite3

db_path = r"D:\Python\Photo_Identify\database\photo_identify_照片.db"

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 清理 face_embeddings 中孤儿
cur.execute("""
DELETE FROM face_embeddings
WHERE image_id NOT IN (SELECT id FROM images);
""")

# 清理 photos 中孤儿
cur.execute("""
DELETE FROM photos
WHERE image_id NOT IN (SELECT id FROM images)
   OR face_id NOT IN (SELECT id FROM face_embeddings)
   OR person_id NOT IN (SELECT id FROM persons);
""")

conn.commit()

# 再次检查外键一致性
cur.execute("PRAGMA foreign_key_check;")
rows = cur.fetchall()
print("foreign_key_check:", rows)

conn.close()
