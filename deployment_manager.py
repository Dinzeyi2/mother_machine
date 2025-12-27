"""
DEPLOYMENT MANAGER - Railway Persistent Services
=================================================

Transforms "Execute-and-Die" into "Deploy-and-Persist"

Features:
1. Creates production service structure (main.py, Dockerfile, requirements.txt)
2. Pushes to GitHub repository
3. Triggers Railway deployment via API
4. Returns permanent URLs
5. Manages service lifecycle
"""

import os
import json
import requests
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import base64
import time

class GitHubManager:
    """Manages GitHub repository operations for deployment"""
    
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = "https://api.github.com"
    
    def create_or_update_repo(self, repo_name: str, description: str = "") -> Dict:
        """Create new repository or get existing one with aggressive sanitization"""
        
        # AGGRESSIVE CLEANING: 
        # 1. Remove all non-printable characters
        # 2. Replace all whitespace (newlines, tabs) with a single space
        # 3. Strip leading/trailing spaces and limit to 255 chars
        clean_description = ""
        if description:
            clean_description = " ".join(description.split())
            clean_description = "".join(char for char in clean_description if char.isprintable())
            clean_description = clean_description[:255]
        
        # Try to get existing repo first
        response = requests.get(
            f"{self.base_url}/repos/{self._get_username()}/{repo_name}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        
        # Create new repo
        response = requests.post(
            f"{self.base_url}/user/repos",
            headers=self.headers,
            json={
                "name": repo_name,
                "description": clean_description,
                "private": False,
                "auto_init": True
            }
        )
        
        if response.status_code == 422:
            error_details = response.json()
            error_msg = error_details.get('errors', [{}])[0].get('message', 'Validation failed')
            raise Exception(f"GitHub Error (422): {error_msg}. Check if '{repo_name}' already exists or contains invalid characters.")
            
        response.raise_for_status()
        return response.json()

    def push_files(self, repo_name: str, files: Dict[str, str], 
                   commit_message: str = "Deploy service") -> Dict:
        """Push multiple files to repository"""
        username = self._get_username()
        ref_response = requests.get(
            f"{self.base_url}/repos/{username}/{repo_name}/git/refs/heads/main",
            headers=self.headers
        )
        ref_response.raise_for_status()
        current_sha = ref_response.json()["object"]["sha"]
        
        commit_response = requests.get(
            f"{self.base_url}/repos/{username}/{repo_name}/git/commits/{current_sha}",
            headers=self.headers
        )
        commit_response.raise_for_status()
        tree_sha = commit_response.json()["tree"]["sha"]
        
        blobs = []
        for file_path, content in files.items():
            blob_response = requests.post(
                f"{self.base_url}/repos/{username}/{repo_name}/git/blobs",
                headers=self.headers,
                json={"content": content, "encoding": "utf-8"}
            )
            blob_response.raise_for_status()
            blobs.append({
                "path": file_path, "mode": "100644", "type": "blob", "sha": blob_response.json()["sha"]
            })
        
        tree_response = requests.post(
            f"{self.base_url}/repos/{username}/{repo_name}/git/trees",
            headers=self.headers,
            json={"base_tree": tree_sha, "tree": blobs}
        )
        tree_response.raise_for_status()
        new_tree_sha = tree_response.json()["sha"]
        
        commit_create_response = requests.post(
            f"{self.base_url}/repos/{username}/{repo_name}/git/commits",
            headers=self.headers,
            json={"message": commit_message, "tree": new_tree_sha, "parents": [current_sha]}
        )
        commit_create_response.raise_for_status()
        new_commit_sha = commit_create_response.json()["sha"]
        
        requests.patch(
            f"{self.base_url}/repos/{username}/{repo_name}/git/refs/heads/main",
            headers=self.headers,
            json={"sha": new_commit_sha}
        ).raise_for_status()
        
        return {
            "commit_sha": new_commit_sha,
            "repo_url": f"https://github.com/{username}/{repo_name}"
        }

    def test_connection(self) -> Dict:
        """Test the GitHub token and return allowed scopes"""
        response = requests.get(f"{self.base_url}/user", headers=self.headers)
        if response.status_code == 401:
             return {"status": "error", "message": "Invalid GITHUB_TOKEN"}
             
        scopes = response.headers.get('X-OAuth-Scopes', '')
        if "repo" not in scopes and "public_repo" not in scopes:
            return {
                "status": "warning",
                "message": "Token is valid but lacks 'repo' scope required for repo creation."
            }
        return {"status": "success", "username": response.json().get("login"), "scopes": scopes}

    def preflight_check(self) -> Dict:
        """Verify GitHub token validity and required 'repo' scope before deployment"""
        try:
            response = requests.get(f"{self.base_url}/user", headers=self.headers)
            if response.status_code == 401:
                return {"status": "error", "message": "Invalid GITHUB_TOKEN. Access denied."}
            
            scopes = response.headers.get('X-OAuth-Scopes', '').split(', ')
            if 'repo' not in scopes:
                return {
                    "status": "warning",
                    "message": "Token valid, but lacks 'repo' scope. Cannot create or push to repositories.",
                    "current_scopes": scopes
                }
            return {
                "status": "success",
                "username": response.json().get("login"),
                "scopes": scopes
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_username(self) -> str:
        """Get authenticated user's username with better error handling"""
        response = requests.get(f"{self.base_url}/user", headers=self.headers)
        if response.status_code == 401:
            raise Exception("GitHub Authentication Failed: The GITHUB_TOKEN is invalid or lacks 'repo' scope.")
        response.raise_for_status()
        return response.json()["login"]

# Ensure you keep RailwayManager, ServiceGenerator, and DeploymentManager below this...
class RailwayManager:
    """Manages Railway deployment operations"""
    
    def __init__(self, api_token: str):
        self.token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://backboard.railway.app/graphql/v2"
    
    def create_project(self, name: str, description: str = "") -> Dict:
        """Create new Railway project"""
        
        mutation = """
        mutation CreateProject($name: String!, $description: String) {
            projectCreate(input: {name: $name, description: $description}) {
                id
                name
            }
        }
        """
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "query": mutation,
                "variables": {
                    "name": name,
                    "description": description
                }
            }
        )
        response.raise_for_status()
        return response.json()["data"]["projectCreate"]
    
    def create_service(self, project_id: str, name: str, 
                       github_repo: str, branch: str = "main") -> Dict:
        """Create service from GitHub repository"""
        
        mutation = """
        mutation CreateService($projectId: String!, $name: String!, 
                              $repo: String!, $branch: String!) {
            serviceCreate(input: {
                projectId: $projectId,
                name: $name,
                source: {
                    repo: $repo,
                    branch: $branch
                }
            }) {
                id
                name
            }
        }
        """
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "query": mutation,
                "variables": {
                    "projectId": project_id,
                    "name": name,
                    "repo": github_repo,
                    "branch": branch
                }
            }
        )
        response.raise_for_status()
        return response.json()["data"]["serviceCreate"]
    
    def get_service_domains(self, service_id: str) -> List[str]:
        """Get all domains for a service"""
        
        query = """
        query GetServiceDomains($serviceId: String!) {
            service(id: $serviceId) {
                domains {
                    domain
                }
            }
        }
        """
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "query": query,
                "variables": {"serviceId": service_id}
            }
        )
        response.raise_for_status()
        
        domains = response.json()["data"]["service"]["domains"]
        return [d["domain"] for d in domains]
    
    def get_deployment_status(self, service_id: str) -> Dict:
        """Get latest deployment status"""
        
        query = """
        query GetDeploymentStatus($serviceId: String!) {
            service(id: $serviceId) {
                deployments {
                    edges {
                        node {
                            id
                            status
                            createdAt
                        }
                    }
                }
            }
        }
        """
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "query": query,
                "variables": {"serviceId": service_id}
            }
        )
        response.raise_for_status()
        
        deployments = response.json()["data"]["service"]["deployments"]["edges"]
        if deployments:
            return deployments[0]["node"]
        return {}
    
    def set_environment_variables(self, service_id: str, variables: Dict[str, str]):
        """Set environment variables for service"""
        
        mutation = """
        mutation SetVariables($serviceId: String!, $variables: [VariableInput!]!) {
            variablesSet(input: {
                serviceId: $serviceId,
                variables: $variables
            })
        }
        """
        
        vars_input = [
            {"name": k, "value": v} for k, v in variables.items()
        ]
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "query": mutation,
                "variables": {
                    "serviceId": service_id,
                    "variables": vars_input
                }
            }
        )
        response.raise_for_status()


class ServiceGenerator:
    """Generates production-ready service files"""
    
    @staticmethod
    def generate_fastapi_service(prompt: str, ai_code: str) -> Dict[str, str]:
        """Generate FastAPI service structure"""
        
        # Extract endpoints from AI-generated code
        endpoints = ServiceGenerator._extract_endpoints(ai_code)
        
        main_py = f'''"""
Auto-generated FastAPI Service
Prompt: {prompt}
Generated: {datetime.now().isoformat()}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Auto-Generated API",
    description="Generated by Mother Machine",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health():
    return {{"status": "healthy", "timestamp": "{datetime.now().isoformat()}"}}

# AI-Generated Code
{ai_code}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
        
        dockerfile = '''FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        requirements = '''fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-multipart==0.0.6
httpx==0.27.2
requests==2.31.0
'''
        
        railway_json = '''{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "./Dockerfile"
  },
  "deploy": {
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10,
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300
  }
}
'''
        
        readme = f'''# Auto-Generated Service

**Generated by Mother Machine**

## Prompt
{prompt}

## Endpoints
{chr(10).join(f"- `{e}`" for e in endpoints)}

## Deployment
Deployed on Railway with automatic CI/CD from GitHub.

## Usage
```bash
curl https://your-service.up.railway.app/health
```
'''
        
        return {
            "main.py": main_py,
            "Dockerfile": dockerfile,
            "requirements.txt": requirements,
            "railway.json": railway_json,
            "README.md": readme,
            ".gitignore": ".env\n__pycache__/\n*.pyc\n.venv/"
        }
    
    @staticmethod
    def _extract_endpoints(code: str) -> List[str]:
        """Extract API endpoints from code"""
        endpoints = []
        
        # Find FastAPI route decorators
        import re
        patterns = [
            r'@app\.(get|post|put|delete|patch)\("([^"]+)"\)',
            r'@app\.(get|post|put|delete|patch)\(\'([^\']+)\'\)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            for method, path in matches:
                endpoints.append(f"{method.upper()} {path}")
        
        return endpoints if endpoints else ["GET /health"]


class DeploymentManager:
    """
    Main deployment orchestrator
    
    Transforms code execution into persistent service deployment
    """
    
    def __init__(self, github_token: str, railway_token: str):
        self.github = GitHubManager(github_token)
        self.railway = RailwayManager(railway_token)
        self.generator = ServiceGenerator()
    
    def deploy_service(self, service_name: str, prompt: str, 
                      ai_generated_code: str, user_id: str) -> Dict:
        """
        Deploy a persistent service
        
        Flow:
        1. Generate service files (main.py, Dockerfile, etc.)
        2. Create/update GitHub repository
        3. Push files to GitHub
        4. Create Railway project
        5. Connect Railway to GitHub
        6. Wait for deployment
        7. Return permanent URLs
        """
        
        deployment_id = f"deploy_{int(time.time())}"
        
        try:
            # STEP 1: Generate service structure
            print(f"[{deployment_id}] Generating service files...")
            files = self.generator.generate_fastapi_service(prompt, ai_generated_code)
            
            # STEP 2: Create GitHub repository
            repo_name = f"{service_name.lower().replace(' ', '-')}-{user_id[:8]}"
            print(f"[{deployment_id}] Creating GitHub repo: {repo_name}")
            
            # FIX: Clean the description to remove control characters/newlines
            clean_description = " ".join(prompt.split())[:1000]

            repo = self.github.create_or_update_repo(
                repo_name=repo_name,
                description=f"Auto-deployed service: {clean_description}"
            )

            # SAFETY CHECK: If repo creation failed, repo might be None
            if not repo or "full_name" not in repo:
                raise Exception(f"GitHub Repository creation failed or returned invalid data for {repo_name}")
            
            # STEP 3: Push files
            print(f"[{deployment_id}] Pushing files to GitHub...")
            push_result = self.github.push_files(
                repo_name=repo_name,
                files=files,
                commit_message=f"Deploy: {prompt}"
            )

            # SAFETY CHECK: If push failed
            if not push_result or "commit_sha" not in push_result:
                raise Exception("Failed to push files to GitHub. Check token permissions.")
            
            # STEP 4: Create Railway project
            print(f"[{deployment_id}] Creating Railway project...")
            project = self.railway.create_project(
                name=repo_name,
                description=f"Service: {service_name}"
            )
            
            # STEP 5: Create service from GitHub
            print(f"[{deployment_id}] Connecting Railway to GitHub...")
            
            # CRITICAL: Verify the repo object exists before accessing "full_name"
            if not repo or "full_name" not in repo:
                raise Exception(f"Cannot connect to Railway: GitHub repository data is missing for {repo_name}")

            service = self.railway.create_service(
                project_id=project["id"],
                name=service_name,
                github_repo=repo["full_name"],
                branch="main"
            )
            
            # STEP 6: Set environment variables
            print(f"[{deployment_id}] Configuring environment...")
            self.railway.set_environment_variables(
                service_id=service["id"],
                variables={
                    "PORT": "8000",
                    "ENVIRONMENT": "production",
                    "DEPLOYED_BY": "mother-machine"
                }
            )
            
            # STEP 7: Wait for deployment (max 5 minutes)
            print(f"[{deployment_id}] Waiting for deployment...")
            max_wait = 300  # 5 minutes
            start_time = time.time()
            
            deployment_status = None
            while time.time() - start_time < max_wait:
                status = self.railway.get_deployment_status(service["id"])
                
                if status.get("status") == "SUCCESS":
                    deployment_status = "deployed"
                    break
                elif status.get("status") == "FAILED":
                    deployment_status = "failed"
                    break
                
                time.sleep(10)
            
            if not deployment_status:
                deployment_status = "deploying"
            
            # STEP 8: Get domains
            domains = self.railway.get_service_domains(service["id"])
            primary_url = f"https://{domains[0]}" if domains else None
            
            return {
                "deployment_id": deployment_id,
                "status": deployment_status,
                "service_name": service_name,
                "service_id": service["id"],
                "project_id": project["id"],
                "github_repo": repo["html_url"],
                "commit_sha": push_result["commit_sha"],
                "url": primary_url,
                "domains": domains,
                "endpoints": self.generator._extract_endpoints(ai_generated_code),
                "deployed_at": datetime.now().isoformat(),
                "user_id": user_id
            }
            
        except Exception as e:
            return {
                "deployment_id": deployment_id,
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
    
    def get_service_status(self, service_id: str) -> Dict:
        """Get current status of deployed service"""
        
        try:
            status = self.railway.get_deployment_status(service_id)
            domains = self.railway.get_service_domains(service_id)
            
            return {
                "service_id": service_id,
                "status": status.get("status", "unknown"),
                "domains": domains,
                "last_deployment": status.get("createdAt"),
                "healthy": status.get("status") == "SUCCESS"
            }
        except Exception as e:
            return {
                "service_id": service_id,
                "status": "error",
                "error": str(e)
            }
    
    def update_service(self, service_id: str, new_code: str, 
                      commit_message: str = "Update service") -> Dict:
        """Update existing service with new code"""
        
        # This would regenerate files and push to GitHub
        # Railway auto-deploys on GitHub push
        pass


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Initialize
    manager = DeploymentManager(
        github_token=os.getenv("GITHUB_TOKEN"),
        railway_token=os.getenv("RAILWAY_TOKEN")
    )
    
    # Deploy a service
    result = manager.deploy_service(
        service_name="Payment API",
        prompt="Build a Stripe payment processing API",
        ai_generated_code='''
@app.post("/payments/create")
async def create_payment(amount: int, currency: str = "usd"):
    return {"payment_id": "pi_123", "status": "succeeded"}

@app.get("/payments/{payment_id}")
async def get_payment(payment_id: str):
    return {"payment_id": payment_id, "status": "succeeded"}
''',
        user_id="user_123"
    )
    
    print(json.dumps(result, indent=2))
    
    if result["status"] == "deployed":
        print(f"\n🎉 Service deployed!")
        print(f"   URL: {result['url']}")
        print(f"   Endpoints: {', '.join(result['endpoints'])}")
