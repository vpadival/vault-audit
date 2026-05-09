from pymongo import MongoClient
from pymongo.database import Database
from faker import Faker
import random
from typing import Any

fake = Faker()
client: MongoClient[dict[str, Any]] = MongoClient("mongodb://localhost:27017/")
db: Database[dict[str, Any]] = client["vault_audit_db"]

# ─── Drop old data on each run (dev convenience) ──────────────────────────────
db.sensitive_data.drop()
db.audit_logs.drop()

# ─── Seed sensitive_data with 10 mock employee records ────────────────────────
departments = ["Engineering", "Finance", "HR", "Legal", "Sales"]
roles       = ["Engineer", "Manager", "Analyst", "Director", "Clerk"]

employees: list[dict[str, Any]] = []
for i in range(10):
    employees.append({
        "employee_id": f"EMP-{1000 + i}",
        "name":        fake.name(),
        "ssn":         fake.ssn(),
        "email":       fake.company_email(),
        "department":  random.choice(departments),
        "role":        random.choice(roles),
        "salary":      random.randint(45_000, 180_000),
        "bank_account": fake.bban(),
    })

db.sensitive_data.insert_many(employees)
print(f"✅  Inserted {len(employees)} employee records into sensitive_data")

# ─── Create audit_logs with schema validation (MongoDB 3.6+) ──────────────────
db.create_collection("audit_logs", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["timestamp", "user_id", "query_type", "record_count",
                     "threat_score", "integrity_hash"],
        "properties": {
            "timestamp":      {"bsonType": "date"},
            "user_id":        {"bsonType": "string"},
            "query_type":     {"bsonType": "string",
                               "enum": ["READ", "UPDATE", "DELETE", "INSERT"]},
            "record_count":   {"bsonType": "int",    "minimum": 0},
            "threat_score":   {"bsonType": "double", "minimum": 0.0, "maximum": 1.0},
            "integrity_hash": {"bsonType": "string"},
            # optional extras captured by the middleware later
            "query_filter":   {"bsonType": "object"},
            "ip_address":     {"bsonType": "string"},
            "flagged":        {"bsonType": "bool"},
        }
    }
})
print("✅  Created audit_logs collection with schema validation")

# ─── Indexes ──────────────────────────────────────────────────────────────────
db.audit_logs.create_index("timestamp")
db.audit_logs.create_index("user_id")
db.audit_logs.create_index("threat_score")
db.sensitive_data.create_index("employee_id", unique=True)
print("✅  Indexes created")

print("\n📦  Collections:", db.list_collection_names())