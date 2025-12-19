"""
Mother Machine Python SDK - Ghost Mode Edition
==============================================

Official Python client with full Ghost Mode support

Usage:
    from mothermachine import MotherMachine
    
    client = MotherMachine(api_key="mm_your_key")
    
    # Build code
    for update in client.build.create(prompt="Build API"):
        print(update)
    
    # Activate Ghost Mode
    job = client.ghost_mode.activate(
        code=my_code,
        aggressive=True
    )
    
    # Check status
    status = client.ghost_mode.get_status(job.job_id)
"""

import requests
import json
import time
from typing import Iterator, List, Dict, Optional
import sseclient


class BuildResponse:
    """Response from build endpoint"""
    def __init__(self, data: Dict):
        self.id = data.get("id")
        self.code = data.get("code")
        self.tests = data.get("tests")
        self.security_score = data.get("security_score")
        self.execution_success = data.get("execution_success")
        self.production_ready = data.get("production_ready")
        self.conversation_id = data.get("conversation_id")


class GhostModeJob:
    """Ghost Mode job"""
    def __init__(self, data: Dict):
        self.job_id = data.get("job_id")
        self.status = data.get("status")
        self.message = data.get("message")
        self.estimated_time = data.get("estimated_time")


class GhostModeStatus:
    """Ghost Mode status"""
    def __init__(self, data: Dict):
        self.job_id = data.get("job_id")
        self.status = data.get("status")
        self.progress = data.get("progress", 0.0)
        self.issues_found = data.get("issues_found", 0)
        self.issues_fixed = data.get("issues_fixed", 0)
        self.test_coverage_before = data.get("test_coverage_before", 0.0)
        self.test_coverage_after = data.get("test_coverage_after", 0.0)
        self.security_score_before = data.get("security_score_before", 0)
        self.security_score_after = data.get("security_score_after", 0)
        self.morning_briefing = data.get("morning_briefing", "")
        self.cost_usd = data.get("cost_usd", 0.0)
        self.error = data.get("error")
    
    @property
    def is_complete(self):
        return self.status == "complete"
    
    @property
    def is_running(self):
        return self.status == "running"
    
    @property
    def has_failed(self):
        return self.status == "failed"


class Build:
    """Build operations"""
    def __init__(self, client):
        self.client = client
    
    def create(self, prompt: str, libraries: Optional[List[str]] = None,
               stream: bool = True, max_healing_attempts: int = 3):
        """Build production code"""
        data = {
            "prompt": prompt,
            "libraries": libraries or [],
            "stream": stream,
            "max_healing_attempts": max_healing_attempts
        }
        
        if stream:
            return self._stream(data)
        else:
            response = self.client._post("/v1/build", data)
            return BuildResponse(response)
    
    def _stream(self, data: Dict) -> Iterator[Dict]:
        """Stream responses"""
        response = requests.post(
            f"{self.client.base_url}/v1/build",
            json=data,
            headers=self.client._headers(),
            stream=True
        )
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break
            yield json.loads(event.data)


class Chat:
    """Chat operations"""
    def __init__(self, client):
        self.client = client
    
    def create(self, messages: List[Dict[str, str]], stream: bool = True):
        """Conversational chat"""
        data = {"messages": messages, "stream": stream}
        
        if stream:
            return self._stream(data)
        else:
            return self.client._post("/v1/chat", data)
    
    def _stream(self, data: Dict) -> Iterator[str]:
        """Stream chat"""
        response = requests.post(
            f"{self.client.base_url}/v1/chat",
            json=data,
            headers=self.client._headers(),
            stream=True
        )
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break
            chunk = json.loads(event.data)
            yield chunk.get("delta", {}).get("content", "")


class GhostMode:
    """
    🌙 Ghost Mode - Overnight autonomy
    
    Activate Ghost Mode to autonomously improve your entire codebase
    """
    def __init__(self, client):
        self.client = client
    
    def activate(self, code: str, aggressive: bool = False,
                 notify_email: Optional[str] = None,
                 notify_webhook: Optional[str] = None) -> GhostModeJob:
        """
        Activate Ghost Mode
        
        Args:
            code: Your code to improve
            aggressive: Fix ALL issues (vs high/medium only)
            notify_email: Email for completion notification
            notify_webhook: Webhook URL for completion
        
        Returns:
            GhostModeJob with job_id and status
        
        Example:
            job = client.ghost_mode.activate(
                code=my_code,
                aggressive=True
            )
            print(f"Ghost Mode activated: {job.job_id}")
        """
        data = {
            "repository_content": code,
            "aggressive": aggressive,
            "notify_email": notify_email,
            "notify_webhook": notify_webhook
        }
        
        response = self.client._post("/v1/ghost-mode", data)
        return GhostModeJob(response)
    
    def get_status(self, job_id: str) -> GhostModeStatus:
        """
        Get Ghost Mode job status
        
        Args:
            job_id: Job ID from activate()
        
        Returns:
            GhostModeStatus with progress and results
        
        Example:
            status = client.ghost_mode.get_status(job.job_id)
            print(f"Progress: {status.progress}%")
        """
        response = self.client._get(f"/v1/ghost-mode/{job_id}")
        return GhostModeStatus(response)
    
    def wait_for_completion(self, job_id: str, poll_interval: int = 30,
                           callback: Optional[callable] = None) -> GhostModeStatus:
        """
        Wait for Ghost Mode to complete
        
        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks (default: 30)
            callback: Optional function called on each status update
        
        Returns:
            Final GhostModeStatus
        
        Example:
            def on_progress(status):
                print(f"Progress: {status.progress}%")
            
            final = client.ghost_mode.wait_for_completion(
                job.job_id,
                callback=on_progress
            )
            print(final.morning_briefing)
        """
        while True:
            status = self.get_status(job_id)
            
            if callback:
                callback(status)
            
            if status.is_complete:
                return status
            
            if status.has_failed:
                raise Exception(f"Ghost Mode failed: {status.error}")
            
            time.sleep(poll_interval)


class MotherMachine:
    """
    Mother Machine API Client - Ghost Mode Edition
    
    Complete production client with:
    - Code building
    - Conversational chat
    - 🌙 Ghost Mode (overnight autonomy)
    
    Example:
        client = MotherMachine(api_key="mm_pro_key_456")
        
        # Build code
        for update in client.build.create(prompt="Build API"):
            print(update)
        
        # Activate Ghost Mode
        job = client.ghost_mode.activate(code=my_code)
        status = client.ghost_mode.wait_for_completion(job.job_id)
        print(status.morning_briefing)
    """
    
    def __init__(self, api_key: str, base_url: str = "https://mothermachine-production.up.railway.app"):
        """
        Initialize Mother Machine client
        
        Args:
            api_key: Your API key (get from mothermachine.ai)
            base_url: API base URL (default: production)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        
        # Initialize sub-clients
        self.build = Build(self)
        self.chat = Chat(self)
        self.ghost_mode = GhostMode(self)
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _post(self, endpoint: str, data: Dict) -> Dict:
        """Make POST request"""
        response = requests.post(
            f"{self.base_url}{endpoint}",
            json=data,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    def _get(self, endpoint: str) -> Dict:
        """Make GET request"""
        response = requests.get(
            f"{self.base_url}{endpoint}",
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_conversation(self, conversation_id: str) -> Dict:
        """Get conversation by ID"""
        return self._get(f"/v1/conversations/{conversation_id}")
    
    def health(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()


# ============================================================
# EXAMPLES
# ============================================================

if __name__ == "__main__":
    # Initialize
    client = MotherMachine(api_key="mm_pro_key_456")
    
    print("🤖 Mother Machine SDK - Ghost Mode Edition\n")
    
    # Example 1: Build code
    print("="*60)
    print("Example 1: Build Payment API")
    print("="*60)
    
    for update in client.build.create(prompt="Build Stripe payment endpoint", stream=True):
        delta = update.get("delta", {})
        if "message" in delta:
            print(f"\n{delta['message']}")
        elif "status" in delta and delta["status"] == "complete":
            print(f"\n✅ Complete!")
            print(f"   Security: {delta['security_score']}/100")
            print(f"   Production ready: {delta['production_ready']}")
    
    # Example 2: Ghost Mode
    print("\n" + "="*60)
    print("Example 2: Ghost Mode (Overnight Autonomy)")
    print("="*60)
    
    sample_code = """
def process_payment(card_number, amount):
    # TODO: Add validation
    # TODO: Add error handling
    return charge_card(card_number, amount)
"""
    
    print("\n🌙 Activating Ghost Mode on sample code...")
    job = client.ghost_mode.activate(code=sample_code, aggressive=True)
    print(f"   Job ID: {job.job_id}")
    print(f"   Status: {job.status}")
    print(f"   {job.message}")
    
    print("\n📊 Checking status...")
    status = client.ghost_mode.get_status(job.job_id)
    print(f"   Status: {status.status}")
    print(f"   Progress: {status.progress}%")
    
    if status.is_complete:
        print("\n🎉 Ghost Mode complete!")
        print(status.morning_briefing)
    else:
        print("\n⏳ Ghost Mode is running... Check back later!")
    
    print("\n" + "="*60)
    print("✅ SDK Examples Complete!")