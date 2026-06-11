#!/usr/bin/env python3
"""
Test script to debug IntercomService fetch_conversations_with_messages
with different last_sync_at dates.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
import json
import aiohttp
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.services.intercom_service import IntercomService
from src.services.token_service import TokenService

class MockTokenService(TokenService):
    """Mock token service that doesn't encrypt/decrypt"""
    def __init__(self):
        # Don't call super().__init__() to avoid encryption setup
        pass
        
    def decrypt_token(self, encrypted_token):
        return encrypted_token  # Return as-is since it's already the API token

async def test_intercom_sync():
    """Test IntercomService with different sync dates"""
    
    # Load environment variables from a local .env file if present
    load_dotenv()

    # Your token (already base64 encoded API token from Intercom)
    # Provide via environment variable INTERCOM_TEST_TOKEN to avoid hardcoding secrets
    test_token = os.getenv("INTERCOM_TEST_TOKEN", "")
    if not test_token:
        raise ValueError("Test token is not set")
    
    # Initialize services with mock token service
    token_service = MockTokenService()
    intercom_service = IntercomService("https://api.intercom.io", token_service)
    
    print("🔍 Testing IntercomService fetch_conversations_with_messages")
    print("=" * 70)
    
    # Test dates
    test_cases = [
        {
            "name": "No last_sync (initial sync)",
            "last_sync_at": None,
            "description": "Should fetch all conversations (historical + new)"
        },
        {
            "name": "January 2025 sync", 
            "last_sync_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "description": "Should fetch conversations updated since Jan 1, 2025"
        },
        {
            "name": "June 2025 sync",
            "last_sync_at": datetime(2025, 6, 1, tzinfo=timezone.utc), 
            "description": "Should fetch conversations updated since June 1, 2025"
        },
        {
            "name": "July 20, 2025 sync",
            "last_sync_at": datetime(2025, 7, 20, tzinfo=timezone.utc),
            "description": "Should fetch conversations updated since July 20, 2025"
        },
        {
            "name": "Future sync", 
            "last_sync_at": datetime(2025, 8, 1, tzinfo=timezone.utc),
            "description": "Should fetch no conversations (future date)"
        }
    ]
    
    project_id = "test-project"
    channels = ["email", "conversation"]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   {test_case['description']}")
        
        if test_case['last_sync_at']:
            print(f"   last_sync_at: {test_case['last_sync_at']}")
            print(f"   timestamp: {test_case['last_sync_at'].timestamp()}")
        else:
            print(f"   last_sync_at: None")
        
        print("   " + "-" * 50)
        
        try:
            conversations = await intercom_service.fetch_conversations_with_messages(
                project_id=project_id,
                encrypted_access_token=test_token,
                channels=channels,
                last_sync_at=test_case['last_sync_at']
            )
            
            print(f"   ✅ SUCCESS: Found {len(conversations)} conversations")
            
            if conversations:
                for j, conv in enumerate(conversations):
                    print(f"      Conversation {j+1}:")
                    print(f"        ID: {conv.source_id}")
                    print(f"        State: {conv.state}")
                    print(f"        Created: {conv.created_at}")
                    print(f"        Updated: {conv.updated_at}")
                    if conv.source:
                        print(f"        Source type: {conv.source.type if conv.source.type else 'None'}")
                    print(f"        Conversation parts: {len(conv.conversation_parts)}")
            else:
                print("      No conversations found")
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("🔍 Testing individual API methods")
    print("=" * 70)
    
    # Test individual methods to see what's happening
    headers = {
        "Authorization": f"Bearer {test_token}",
        "Accept": "application/json"
    }
    
    # Test _search_conversations_since_date with January date
    january_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    print(f"\n🔍 Testing _search_conversations_since_date with {january_date}")
    print(f"   Timestamp: {january_date.timestamp()}")
    
    try:
        # Manually call the method to debug
        since_conversations = await intercom_service._search_conversations_since_date(
            headers, channels, january_date, 100
        )
        print(f"   ✅ _search_conversations_since_date returned {len(since_conversations)} conversations")
        
        if since_conversations:
            for conv in since_conversations:
                print(f"      ID: {conv.get('id')}")
                print(f"      State: {conv.get('state')}")
                print(f"      Created: {datetime.fromtimestamp(conv.get('created_at', 0))}")
                print(f"      Updated: {datetime.fromtimestamp(conv.get('updated_at', 0))}")
        
    except Exception as e:
        print(f"   ❌ ERROR in _search_conversations_since_date: {str(e)}")
        import traceback
        traceback.print_exc()

    # Test raw API call to compare
    print(f"\n🔍 Testing raw API call for comparison")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": {
                    "operator": "AND",
                    "value": [
                        {"field": "state", "operator": "=", "value": "closed"},
                        {"field": "updated_at", "operator": ">", "value": january_date.timestamp()}
                    ]
                },
                "pagination": {"per_page": 10}
            }
            
            async with session.post(
                "https://api.intercom.io/conversations/search",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                conversations = data.get("conversations", [])
                print(f"   ✅ Raw API call returned {len(conversations)} conversations")
                
                if conversations:
                    for conv in conversations:
                        print(f"      ID: {conv.get('id')}")
                        print(f"      State: {conv.get('state')}")
                        print(f"      Updated: {datetime.fromtimestamp(conv.get('updated_at', 0))}")
                        
    except Exception as e:
        print(f"   ❌ ERROR in raw API call: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_intercom_sync()) 