"""Quick Supabase connection and table check."""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")

if not url or url.startswith("your_"):
    print("❌ SUPABASE_URL not set in .env")
    sys.exit(1)

print(f"✅ Supabase URL loaded: {url}")
client = create_client(url, key)
print("✅ Supabase client created\n")

for table in ["leads", "handoffs"]:
    try:
        client.table(table).select("id").limit(1).execute()
        print(f"✅ Table '{table}': EXISTS")
    except Exception as e:
        err = str(e)
        if "does not exist" in err or "42P01" in err or "relation" in err.lower():
            print(f"❌ Table '{table}': NOT FOUND — needs to be created via SQL Editor")
        else:
            print(f"⚠️  Table '{table}': ERROR — {err[:120]}")
