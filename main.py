"""
MOTHER MACHINE - ULTIMATE EDITION WITH PERSISTENT DEPLOYMENT
=============================================================

ALL FEATURES IN ONE FILE:
1. Smart Intent Routing (Chain-of-Thought)
2. Persistent Memory (PostgreSQL/SQLite)
3. v5 Sandbox (Real Execution + Self-Healing)
4. Ghost Mode (Overnight Autonomy)
5. REAL Autonomous Coding (72+ hours, Git, Multi-file)
6. 🚀 NEW: Persistent Service Deployment (Deploy-and-Persist)
7. Research-Backed

For: https://github.com/Dinzeyi2/mother_machine
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import os, sys, json, tempfile, subprocess, time, re, uuid
from datetime import datetime
from pathlib import Path
import sqlite3

# Database imports
try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

from anthropic import Anthropic

# IMPORT AUTONOMOUS ENGINE, DEPLOYMENT MANAGER, AND LIFECYCLE MANAGER
sys.path.append(os.path.dirname(__file__))
from autonomous_engine import AutonomousCodingEngine
from deployment_manager import DeploymentManager
from lifecycle_manager import DeploymentTracker, LifecycleManager, DeploymentProvider

# Initialize
app = FastAPI(
    title="Mother Machine - Ultimate Edition with Deployment",
    description="Complete AI software engineering with persistent service deployment",
    version="8.0.0-deployment"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize deployment manager and lifecycle tracker
deployment_tracker = DeploymentTracker()

try:
    deployment_manager = DeploymentManager(
        github_token=os.getenv("GITHUB_TOKEN"),
        railway_token=os.getenv("RAILWAY_TOKEN")
    )
    lifecycle_manager = LifecycleManager(
        railway_token=os.getenv("RAILWAY_TOKEN"),
        tracker=deployment_tracker
    )
    DEPLOYMENT_ENABLED = True
except Exception as e:
    print(f"⚠️  Deployment disabled: {e}")
    DEPLOYMENT_ENABLED = False

# ============================================================
# PERSISTENT MEMORY (Existing)
# ============================================================

class PersistentMemory:
    """Permanent user memory (survives restarts)"""
    
    def __init__(self):
        self.use_postgres = os.getenv('DATABASE_URL') is not None and HAS_POSTGRES
        
        if self.use_postgres:
            self._init_postgres()
        else:
            self._init_sqlite()
    
    def _init_postgres(self):
        """Initialize PostgreSQL"""
        self.conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_contexts (
                    user_id TEXT PRIMARY KEY,
                    conversation_history JSONB,
                    current_project TEXT,
                    preferences JSONB,
                    last_intent TEXT,
                    session_started TIMESTAMP,
                    domain_expertise JSONB,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            self.conn.commit()
    
    def _init_sqlite(self):
        """Initialize SQLite (local dev)"""
        self.conn = sqlite3.connect('user_memory.db', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_contexts (
                user_id TEXT PRIMARY KEY,
                conversation_history TEXT,
                current_project TEXT,
                preferences TEXT,
                last_intent TEXT,
                session_started TEXT,
                domain_expertise TEXT,
                updated_at TEXT
            )
        """)
        self.conn.commit()
    
    def save(self, user_id: str, data: dict):
        """Save user context"""
        if self.use_postgres:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_contexts (user_id, conversation_history, current_project,
                                              preferences, last_intent, session_started,
                                              domain_expertise, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        conversation_history = EXCLUDED.conversation_history,
                        current_project = EXCLUDED.current_project,
                        preferences = EXCLUDED.preferences,
                        last_intent = EXCLUDED.last_intent,
                        domain_expertise = EXCLUDED.domain_expertise,
                        updated_at = NOW()
                """, (
                    user_id,
                    Json(data.get('conversation_history', [])),
                    data.get('current_project'),
                    Json(data.get('preferences', {})),
                    data.get('last_intent'),
                    data.get('session_started'),
                    Json(data.get('domain_expertise', {}))
                ))
                self.conn.commit()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO user_contexts
                (user_id, conversation_history, current_project, preferences,
                 last_intent, session_started, domain_expertise, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                json.dumps(data.get('conversation_history', [])),
                data.get('current_project'),
                json.dumps(data.get('preferences', {})),
                data.get('last_intent'),
                data.get('session_started'),
                json.dumps(data.get('domain_expertise', {})),
                datetime.now().isoformat()
            ))
            self.conn.commit()
    
    def load(self, user_id: str) -> Optional[dict]:
        """Load user context"""
        if self.use_postgres:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM user_contexts WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return dict(row)
        else:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM user_contexts WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                'user_id': row['user_id'],
                'conversation_history': json.loads(row['conversation_history']),
                'current_project': row['current_project'],
                'preferences': json.loads(row['preferences']),
                'last_intent': row['last_intent'],
                'session_started': row['session_started'],
                'domain_expertise': json.loads(row['domain_expertise'])
            }

memory = PersistentMemory()

# ============================================================
# MODELS
# ============================================================

class ExecuteRequest(BaseModel):
    code: str
    input: Optional[str] = ""
    language: Optional[str] = "python"
    timeout: Optional[int] = 30000

class DeployRequest(BaseModel):
    """Deploy a persistent service - THE NEW WAY"""
    service_name: str = Field(..., description="Name of the service")
    prompt: str = Field(..., description="What the service should do")
    user_id: str = Field(..., description="User ID")
    environment_vars: Optional[dict] = {}

class ServiceUpdateRequest(BaseModel):
    code: str
    commit_message: Optional[str] = "Update service"

class SmartRequest(BaseModel):
    message: str
    user_id: str
    stream: Optional[bool] = True

class BuildRequest(BaseModel):
    prompt: str
    user_id: str
    stream: Optional[bool] = True

class GhostModeRequest(BaseModel):
    repository_content: str
    user_id: str
    aggressive: Optional[bool] = False

class AutonomousReq(BaseModel):
    task: str
    repository_url: str
    github_token: str
    user_id: str
    max_hours: Optional[int] = 72

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_user_context(user_id: str) -> dict:
    """Get or create user context"""
    ctx = memory.load(user_id)
    if ctx:
        return ctx
    
    return {
        'user_id': user_id,
        'conversation_history': [],
        'current_project': None,
        'preferences': {},
        'last_intent': None,
        'session_started': datetime.now().isoformat(),
        'domain_expertise': {}
    }

def update_user_context(user_id: str, message: str, intent: str):
    """Update and save user context"""
    ctx = get_user_context(user_id)
    ctx['conversation_history'].append({
        'role': 'user',
        'content': message,
        'intent': intent,
        'timestamp': datetime.now().isoformat()
    })
    ctx['last_intent'] = intent
    
    domains = ['finance', 'health', 'legal', 'ecommerce']
    for domain in domains:
        if domain in message.lower():
            ctx['domain_expertise'][domain] = ctx['domain_expertise'].get(domain, 0) + 1
    
    memory.save(user_id, ctx)
    return ctx

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    features = [
        "✅ Smart Intent Routing",
        "✅ Persistent Memory",
        "✅ v5 Sandbox (Execute-and-Die for scripts)",
        "✅ Ghost Mode",
        "✅ REAL Autonomous Coding (72+ hours)",
        "✅ Git Integration"
    ]
    
    if DEPLOYMENT_ENABLED:
        features.append("✅ 🚀 Persistent Service Deployment (Deploy-and-Persist)")
    
    return {
        "service": "Mother Machine - Ultimate Edition with Deployment",
        "version": "8.0.0-deployment",
        "features": features,
        "deployment_enabled": DEPLOYMENT_ENABLED,
        "github": "https://github.com/Dinzeyi2/mother_machine"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory": "PostgreSQL" if memory.use_postgres else "SQLite",
        "deployment": "enabled" if DEPLOYMENT_ENABLED else "disabled"
    }

# ============================================================
# EXECUTE-AND-DIE (For Scripts/Tests)
# ============================================================

@app.post("/v1/execute")
async def execute_code(r: ExecuteRequest):
    """
    Execute-and-Die Model
    
    Good for:
    - Quick scripts
    - Testing
    - Data processing
    - One-off calculations
    
    NOT good for:
    - APIs that need URLs
    - Services that need to stay running
    - Anything that needs persistent state
    
    Use /v1/deploy for those instead!
    """
    import subprocess, tempfile, time
    from pathlib import Path
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "code.py"
            code_file.write_text(r.code)
            
            start_time = time.time()
            
            process = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=(r.timeout / 1000.0),
                cwd=tmpdir,
                input=r.input if r.input else None
            )
            
            return {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "exitCode": process.returncode,
                "executionTime": f"{time.time() - start_time:.3f}s",
                "model": "execute-and-die",
                "note": "Container dies after this. Use /v1/deploy for persistent services."
            }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Execution Timeout",
            "exitCode": 124,
            "model": "execute-and-die"
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exitCode": 1,
            "model": "execute-and-die"
        }

# ============================================================
# DEPLOY-AND-PERSIST (For Production Services)
# ============================================================

@app.post("/v1/deploy")
async def deploy_persistent_service(request: DeployRequest, bg: BackgroundTasks):
    """
    🚀 Deploy-and-Persist Model
    
    THIS IS THE KEY DIFFERENCE FROM /v1/execute!
    
    /v1/execute:
    - Runs code ONCE
    - Returns output
    - Container dies
    - No URL
    - Can't receive requests
    
    /v1/deploy:
    - Generates full service structure
    - Pushes to GitHub
    - Deploys to Railway
    - Gets permanent URL
    - Stays alive 24/7
    - Handles unlimited requests
    
    Perfect for:
    - REST APIs
    - Web services
    - Webhooks
    - Background workers
    - Real-time applications
    
    Example:
    ```
    POST /v1/deploy
    {
        "service_name": "Payment API",
        "prompt": "Build Stripe payment processing endpoints",
        "user_id": "user_123"
    }
    
    Response:
    {
        "deployment_id": "deploy_1234",
        "status": "deploying",
        "url": "https://payment-api-user123.up.railway.app",
        "endpoints": ["POST /payments/create", "GET /payments/{id}"]
    }
    ```
    
    The deployed service:
    - Has its own URL
    - Runs 24/7
    - Auto-restarts on failure
    - Auto-deploys on GitHub push
    """
    
    if not DEPLOYMENT_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Deployment disabled. Set GITHUB_TOKEN and RAILWAY_TOKEN environment variables."
        )
    
    # Generate code using AI
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""Create production FastAPI endpoints for: {request.prompt}

Requirements:
1. Use FastAPI decorators (@app.get, @app.post, etc.)
2. Include proper error handling
3. Add request/response models with Pydantic
4. Include docstrings
5. Make it production-ready
6. Add proper CORS headers
7. Include health check endpoint

Generate ONLY the endpoint code (no imports, no app creation).

Example format:
```python
@app.post("/resource")
async def create_resource(data: dict):
    # Your code
    return {{"id": 123, "status": "created"}}
```
"""
        }]
    )
    
    ai_code = response.content[0].text
    
    # Generate deployment ID
    deployment_id = f"deploy_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Save initial status
    memory.save(deployment_id, {
        "deployment_id": deployment_id,
        "status": "queued",
        "service_name": request.service_name,
        "prompt": request.prompt,
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat()
    })
    
    # Deploy in background
    bg.add_task(
        deploy_service_background,
        deployment_id,
        request.service_name,
        request.prompt,
        ai_code,
        request.user_id,
        request.environment_vars
    )
    
    return {
        "deployment_id": deployment_id,
        "status": "deploying",
        "message": "🚀 Deploying persistent service...",
        "eta_minutes": 3,
        "check_status": f"/v1/deployments/{deployment_id}",
        "how_it_works": {
            "step_1": "Generate service files (main.py, Dockerfile, etc.)",
            "step_2": "Push to GitHub repository",
            "step_3": "Railway builds Docker image",
            "step_4": "Deploy and assign permanent URL",
            "step_5": "Service runs 24/7 forever"
        },
        "difference_from_execute": {
            "execute": "Runs once, returns output, dies ☠️",
            "deploy": "Runs forever, gets URL, handles unlimited requests ♾️"
        }
    }

def deploy_service_background(deployment_id: str, service_name: str,
                              prompt: str, ai_code: str, user_id: str,
                              env_vars: dict):
    """Background deployment task with lifecycle tracking"""
    try:
        # STEP 1: Create tracking record
        deployment_tracker.create_deployment(
            deployment_id=deployment_id,
            user_id=user_id,
            service_name=service_name,
            prompt=prompt,
            provider=DeploymentProvider.RAILWAY
        )
        
        # Update status
        memory.save(deployment_id, {
            "deployment_id": deployment_id,
            "status": "deploying",
            "phase": "generating_files",
            "message": "Generating service structure..."
        })
        
        # STEP 2: Deploy using deployment manager
        result = deployment_manager.deploy_service(
            service_name=service_name,
            prompt=prompt,
            ai_generated_code=ai_code,
            user_id=user_id
        )
        
        # STEP 3: Update external IDs in tracker
        if result.get('status') == 'deployed':
            deployment_tracker.update_external_ids(
                deployment_id=deployment_id,
                project_id=result.get('project_id'),
                service_id=result.get('service_id'),
                deployment_ext_id=None
            )
            
            # STEP 4: Mark as deployed
            deployment_tracker.mark_deployed(
                deployment_id=deployment_id,
                service_url=result.get('url'),
                github_repo_url=result.get('github_repo'),
                endpoints=result.get('endpoints', [])
            )
        else:
            # Mark as failed
            deployment_tracker.mark_failed(
                deployment_id=deployment_id,
                error=result.get('error', 'Unknown error')
            )
        
        # Save result
        result["deployment_id"] = deployment_id
        memory.save(deployment_id, result)
        
    except Exception as e:
        deployment_tracker.mark_failed(deployment_id, str(e))
        memory.save(deployment_id, {
            "deployment_id": deployment_id,
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

# ============================================================
# LIFECYCLE MANAGEMENT ENDPOINTS
# ============================================================

@app.delete("/v1/deployments/{deployment_id}")
async def delete_deployment(deployment_id: str):
    """
    🗑️ DELETE DEPLOYMENT - PREVENTS ORPHANED SERVICES
    
    This is CRITICAL! Without this, you keep paying for deleted services.
    
    What it does:
    1. Deletes service from Railway
    2. Deletes project from Railway
    3. Marks as deleted in database
    4. Stops charging you
    
    Without lifecycle management:
    - User deletes in your app ✅
    - Railway service still running 💸
    - You keep paying ❌
    
    With lifecycle management:
    - User deletes in your app ✅
    - Railway service deleted ✅
    - Database updated ✅
    - No more charges ✅
    """
    
    if not DEPLOYMENT_ENABLED:
        raise HTTPException(503, "Deployment disabled")
    
    result = lifecycle_manager.delete_deployment(deployment_id)
    
    if not result['success']:
        raise HTTPException(400, result.get('error', 'Deletion failed'))
    
    return {
        "success": True,
        "deployment_id": deployment_id,
        "message": "Deployment deleted from Railway and database",
        "details": result
    }

@app.get("/v1/deployments/user/{user_id}")
async def list_user_deployments(user_id: str, status: Optional[str] = None):
    """
    📋 LIST USER'S DEPLOYMENTS
    
    Shows all deployments for a user with their status and costs.
    """
    
    from lifecycle_manager import DeploymentStatus
    
    status_filter = None
    if status:
        try:
            status_filter = DeploymentStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")
    
    deployments = deployment_tracker.get_user_deployments(
        user_id=user_id,
        status=status_filter
    )
    
    return {
        "user_id": user_id,
        "total": len(deployments),
        "deployments": deployments
    }

@app.get("/v1/deployments/costs/{user_id}")
async def get_user_costs(user_id: str):
    """
    💰 GET USER'S DEPLOYMENT COSTS
    
    Shows:
    - Total deployments
    - Active deployments (costing money)
    - Monthly cost
    - Total spent
    
    Use this to show users their spending!
    """
    
    costs = deployment_tracker.calculate_costs(user_id)
    
    return {
        "user_id": user_id,
        "total_deployments": costs['total_deployments'],
        "active_deployments": costs['active_deployments'],
        "monthly_cost_usd": float(costs['monthly_cost'] or 0),
        "total_spent_usd": float(costs['total_spent'] or 0),
        "cost_breakdown": {
            "railway_per_service": 5.0,
            "estimated_monthly": float(costs['monthly_cost'] or 0)
        }
    }

@app.post("/v1/deployments/cleanup/orphans")
async def cleanup_orphaned_deployments(user_id: Optional[str] = None):
    """
    🧹 CLEANUP ORPHANED DEPLOYMENTS
    
    Finds services that are:
    - Marked as ACTIVE in database
    - But don't exist in Railway anymore
    
    Run this periodically to prevent database/Railway desync.
    
    Without this, your database shows services that don't exist,
    and you can't properly track costs.
    """
    
    if not DEPLOYMENT_ENABLED:
        raise HTTPException(503, "Deployment disabled")
    
    result = lifecycle_manager.cleanup_orphans(user_id)
    
    return {
        "total_checked": result['total_checked'],
        "deleted": result['deleted'],
        "failed": result['failed'],
        "message": f"Cleaned up {len(result['deleted'])} orphaned deployments"
    }

@app.get("/v1/deployments/orphans/{user_id}")
async def find_orphaned_deployments(user_id: str):
    """
    🔍 FIND POTENTIAL ORPHANS
    
    Shows deployments that MIGHT be orphaned.
    Review these before running cleanup.
    """
    
    all_active = deployment_tracker.get_user_deployments(
        user_id=user_id,
        status=DeploymentStatus.ACTIVE
    )
    
    return {
        "user_id": user_id,
        "potentially_orphaned": len(all_active),
        "deployments": all_active,
        "warning": "These are marked ACTIVE in database. Verify they exist in Railway."
    }

@app.get("/v1/deployments/{deployment_id}")
async def get_deployment_status(deployment_id: str):
    """
    Get deployment status
    
    Statuses:
    - queued: Waiting to start
    - deploying: Building and deploying
    - deployed: Live and ready! 🎉
    - failed: Deployment failed ❌
    """
    
    result = memory.load(deployment_id)
    
    if not result:
        raise HTTPException(404, "Deployment not found")
    
    return result

@app.get("/v1/services/{service_id}")
async def get_service_status(service_id: str):
    """
    Get running service health
    
    Checks if the PERSISTENT service is:
    - Running
    - Responding to health checks
    - Has recent successful deployments
    """
    
    if not DEPLOYMENT_ENABLED:
        raise HTTPException(503, "Deployment disabled")
    
    status = deployment_manager.get_service_status(service_id)
    return status

@app.put("/v1/services/{service_id}")
async def update_service(service_id: str, request: ServiceUpdateRequest):
    """
    Update running service with new code
    
    This pushes new code to GitHub, which triggers Railway
    to auto-deploy the update WITHOUT downtime.
    
    This is continuous deployment in action!
    """
    
    if not DEPLOYMENT_ENABLED:
        raise HTTPException(503, "Deployment disabled")
    
    result = deployment_manager.update_service(
        service_id=service_id,
        new_code=request.code,
        commit_message=request.commit_message
    )
    
    return {
        "service_id": service_id,
        "status": "updating",
        "message": "Code pushed to GitHub, Railway will auto-deploy",
        "eta_minutes": 2
    }

@app.get("/v1/compare")
async def compare_execute_vs_deploy():
    """
    Understand the difference between Execute and Deploy
    """
    
    return {
        "execute_and_die": {
            "endpoint": "/v1/execute",
            "lifecycle": "Transient (seconds)",
            "use_cases": [
                "Running tests",
                "Data processing",
                "Scripts",
                "One-off calculations"
            ],
            "limitations": [
                "No persistent state",
                "No URL",
                "Can't receive HTTP requests",
                "Memory clears after execution"
            ],
            "example": {
                "request": {"code": "print('hello')"},
                "response": {"stdout": "hello", "exitCode": 0},
                "then": "Container dies ☠️"
            }
        },
        "deploy_persistent": {
            "endpoint": "/v1/deploy",
            "lifecycle": "Persistent (24/7)",
            "use_cases": [
                "REST APIs",
                "Web services",
                "Webhooks",
                "Real-time applications",
                "Background workers"
            ],
            "benefits": [
                "Permanent URL",
                "Handles unlimited requests",
                "Keeps state in memory/database",
                "Auto-restarts on failure",
                "Auto-deploys on code push"
            ],
            "example": {
                "request": {
                    "service_name": "Payment API",
                    "prompt": "Build Stripe payment endpoints"
                },
                "response": {
                    "url": "https://payment-api.up.railway.app",
                    "endpoints": ["POST /payments/create"]
                },
                "then": "Service runs forever ♾️"
            }
        },
        "key_difference": {
            "execute": "Lambda-style: Run → Return → Die",
            "deploy": "Server-style: Deploy → Run Forever → Serve Requests"
        }
    }

# ============================================================
# EXISTING ENDPOINTS (Autonomous, Ghost Mode, etc.)
# ============================================================

@app.post("/v1/autonomous")
async def autonomous(r: AutonomousReq, bg: BackgroundTasks):
    """REAL autonomous coding - codes for days"""
    jid = f"job_{uuid.uuid4().hex[:8]}"
    
    memory.save(jid, {
        "status": "queued",
        "task": r.task,
        "repo": r.repository_url,
        "user": r.user_id,
        "created": datetime.now().isoformat()
    })
    
    bg.add_task(
        run_autonomous_job,
        jid, r.task, r.repository_url, r.github_token, r.max_hours, r.user_id
    )
    
    return {
        "job_id": jid,
        "status": "started",
        "check_status": f"/v1/jobs/{jid}"
    }

def run_autonomous_job(job_id, task, repo_url, github_token, max_hours, user_id):
    """Background autonomous job"""
    try:
        engine = AutonomousCodingEngine(job_id, memory)
        engine.run(
            task=task,
            repo_url=repo_url,
            github_token=github_token,
            max_hours=max_hours,
            user_id=user_id
        )
    except Exception as e:
        memory.save(job_id, {
            "status": "failed",
            "error": str(e)
        })

@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status"""
    j = memory.load(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():
    db_type = "PostgreSQL" if memory.use_postgres else "SQLite"
    print("\n" + "="*70)
    print("🔥 MOTHER MACHINE - ULTIMATE EDITION WITH DEPLOYMENT")
    print("="*70)
    print(f"\n🧠 Features:")
    print(f"   ✅ Execute-and-Die (Scripts/Tests)")
    if DEPLOYMENT_ENABLED:
        print(f"   ✅ 🚀 Deploy-and-Persist (Production Services)")
    else:
        print(f"   ⚠️  Deploy-and-Persist (Disabled - set tokens)")
    print(f"   ✅ Persistent Memory ({db_type})")
    print(f"   ✅ Autonomous Coding (72+ hours)")
    print(f"\n📡 Key Endpoints:")
    print(f"   POST /v1/execute - Run code once (execute-and-die)")
    if DEPLOYMENT_ENABLED:
        print(f"   POST /v1/deploy - Deploy service forever (deploy-and-persist)")
    print(f"   GET /v1/compare - Understand the difference")
    print("="*70 + "\n")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

