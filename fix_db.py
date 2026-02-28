import sqlite3
import os

DB_PATH = 'fake_news.db'

def fix_database():
    if not os.path.exists(DB_PATH):
        print("Database file not found. It will be created when you run app.py.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Checking 'predictions' table schema...")
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'content_type' not in columns:
        print("Migrating 'predictions' table to new schema...")
        
        # 1. Rename existing table to a backup
        cursor.execute("ALTER TABLE predictions RENAME TO predictions_old")
        
        # 2. Create the new table
        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                content_type TEXT,
                input_content TEXT,
                prediction_result TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        
        # 3. Attempt to migrate old data if possible
        # Old schema: id, user_id, real_news, fake_news, timestamp
        try:
            cursor.execute('''
                INSERT INTO predictions (id, user_id, content_type, input_content, prediction_result, confidence, timestamp)
                SELECT id, user_id, 'text', COALESCE(real_news, fake_news), 
                CASE WHEN fake_news IS NOT NULL THEN 'fake' ELSE 'real' END, 
                0.9, timestamp FROM predictions_old
            ''')
            print("Successfully migrated old data.")
        except Exception as e:
            print(f"No data to migrate or error: {e}")
            
        # 4. Drop the old table
        cursor.execute("DROP TABLE predictions_old")
        
        print("\x1b[32mDatabase migration complete!\x1b[0m")
    else:
        print("Database schema is already up to date.")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    fix_database()
