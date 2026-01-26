"""Test script to check if appointments are in vector DB"""
from calendar_agent.weaviate_db import get_weaviate_client
from calendar_agent.sql_db import get_db_connection
from psycopg2.extras import RealDictCursor

def check_vector_db_data():
    """Check what's in the vector database"""
    client = get_weaviate_client()
    
    if not client:
        print("❌ Cannot connect to Weaviate")
        return
    
    try:
        appointment_collection = client.collections.get("AppointmentHistory")
        
        # Get all appointments in vector DB
        response = appointment_collection.query.fetch_objects(limit=100)
        
        print(f"\n{'='*60}")
        print(f"VECTOR DB STATUS")
        print(f"{'='*60}")
        print(f"Total appointments in vector DB: {len(response.objects)}\n")
        
        if response.objects:
            print("Sample appointments in vector DB:")
            for i, obj in enumerate(response.objects[:5], 1):
                props = obj.properties
                print(f"\n{i}. ID: {props['appointment_id']}")
                print(f"   Customer: {props['customer_name']}")
                print(f"   Title: {props['title']}")
                print(f"   Description: {props['description'][:80]}...")
        else:
            print("⚠️ No appointments found in vector DB!")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error checking vector DB: {e}")

def check_sql_db_data():
    """Check what's in the SQL database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT COUNT(*) as count FROM appointments")
        result = cur.fetchone()
        total_sql = result['count']
        
        cur.execute("""
            SELECT id, customer_name, appointment_title, notes, address
            FROM appointments
            ORDER BY start_time DESC
            LIMIT 5
        """)
        appointments = cur.fetchall()
        
        cur.close()
        conn.close()
        
        print(f"{'='*60}")
        print(f"SQL DB STATUS")
        print(f"{'='*60}")
        print(f"Total appointments in SQL DB: {total_sql}\n")
        
        if appointments:
            print("Recent appointments in SQL DB:")
            for i, apt in enumerate(appointments, 1):
                print(f"\n{i}. ID: {apt['id']}")
                print(f"   Customer: {apt['customer_name']}")
                print(f"   Title: {apt['appointment_title']}")
                notes = apt['notes'] or ""
                if notes:
                    print(f"   Notes: {notes[:80]}...")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error checking SQL DB: {e}")

def test_search():
    """Test a sample search"""
    print(f"{'='*60}")
    print(f"TESTING SEARCH")
    print(f"{'='*60}\n")
    
    from calendar_agent.weaviate_db import find_related_appointments
    
    # Test 1: Search with actual customer
    print("TEST 1: Searching for roof work for 'Lee Deffebach'...")
    results = find_related_appointments(
        customer_name="Lee Deffebach",
        title="Roof shingle repair",
        description="Need to repair damaged shingles on the roof",
        address="123 Main St",
        limit=5,
        distance_threshold=0.8
    )
    
    print(f"Found {len(results)} related appointments")
    for i, apt in enumerate(results, 1):
        print(f"\n{i}. ID: {apt['appointment_id']}")
        print(f"   Title: {apt['title']}")
        print(f"   Description: {apt['description'][:60]}...")
        if apt['distance'] is not None:
            print(f"   Distance: {apt['distance']:.3f} (lower is more similar)")
            print(f"   Similarity: {apt['similarity_score']:.2%}")
        else:
            print(f"   Distance: N/A")
            print(f"   Similarity: N/A")
    
    # Test 2: Search for plumbing work
    print(f"\n\nTEST 2: Searching for plumbing work for 'Lee Deffebach'...")
    results2 = find_related_appointments(
        customer_name="Lee Deffebach",
        title="Bathroom sink repair",
        description="Fix leaking bathroom sink faucet",
        address="456 Oak St",
        limit=5,
        distance_threshold=0.8
    )
    
    print(f"Found {len(results2)} related appointments")
    for i, apt in enumerate(results2, 1):
        print(f"\n{i}. ID: {apt['appointment_id']}")
        print(f"   Title: {apt['title']}")
        print(f"   Description: {apt['description'][:60]}...")
        if apt['distance'] is not None:
            print(f"   Distance: {apt['distance']:.3f}")
            print(f"   Similarity: {apt['similarity_score']:.2%}")
        else:
            print(f"   Distance: N/A")
            print(f"   Similarity: N/A")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    check_sql_db_data()
    check_vector_db_data()
    test_search()
