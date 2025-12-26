"""
LIFECYCLE MANAGEMENT SYSTEM
============================

Tracks ALL deployed services and syncs with Railway.
Prevents orphaned services that cost money.

Features:
1. Database tracking of all deployments
2. Create → Track → Delete sync
3. Cost monitoring
4. Orphan detection
5. Automatic cleanup
"""

import os
import json
import sqlite3
import requests
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False


class DeploymentProvider(str, Enum):
    RAILWAY = "railway"
    MODAL = "modal"
    FLY_IO = "flyio"
    LOCAL = "local"


class DeploymentStatus(str, Enum):
    CREATING = "creating"
    ACTIVE = "active"
    UPDATING = "updating"
    DELETING = "deleting"
    DELETED = "deleted"
    FAILED = "failed"


class DeploymentTracker:
    """
    Tracks ALL deployments across providers
    
    Critical table: deployments
    - Stores EVERY service we create
    - Links to external provider IDs
    - Enables cleanup when user deletes
    """
    
    def __init__(self):
        self.use_postgres = os.getenv('DATABASE_URL') is not None and HAS_POSTGRES
        
        if self.use_postgres:
            self._init_postgres()
        else:
            self._init_sqlite()
    
    def _init_postgres(self):
        """PostgreSQL schema"""
        self.conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        with self.conn.cursor() as cur:
            # Main deployments table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id SERIAL PRIMARY KEY,
                    deployment_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    service_name TEXT NOT NULL,
                    prompt TEXT,
                    
                    -- Provider info
                    provider TEXT NOT NULL,
                    external_project_id TEXT,
                    external_service_id TEXT,
                    external_deployment_id TEXT,
                    
                    -- URLs
                    service_url TEXT,
                    github_repo_url TEXT,
                    
                    -- Status
                    status TEXT NOT NULL,
                    error_message TEXT,
                    
                    -- Metadata
                    endpoints JSONB,
                    environment_vars JSONB,
                    
                    -- Costs (estimated)
                    monthly_cost_usd DECIMAL(10, 2),
                    total_cost_usd DECIMAL(10, 2) DEFAULT 0,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    deployed_at TIMESTAMP,
                    deleted_at TIMESTAMP,
                    
                    -- Indexes
                    INDEX idx_user_id (user_id),
                    INDEX idx_deployment_id (deployment_id),
                    INDEX idx_status (status),
                    INDEX idx_provider (provider),
                    INDEX idx_external_service_id (external_service_id)
                )
            """)
            
            # Usage tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS deployment_usage (
                    id SERIAL PRIMARY KEY,
                    deployment_id TEXT REFERENCES deployments(deployment_id),
                    date DATE NOT NULL,
                    requests_count INTEGER DEFAULT 0,
                    uptime_hours DECIMAL(10, 2) DEFAULT 0,
                    cost_usd DECIMAL(10, 2) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(deployment_id, date)
                )
            """)
            
            self.conn.commit()
    
    def _init_sqlite(self):
        """SQLite schema"""
        self.conn = sqlite3.connect('deployments.db', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                service_name TEXT NOT NULL,
                prompt TEXT,
                
                provider TEXT NOT NULL,
                external_project_id TEXT,
                external_service_id TEXT,
                external_deployment_id TEXT,
                
                service_url TEXT,
                github_repo_url TEXT,
                
                status TEXT NOT NULL,
                error_message TEXT,
                
                endpoints TEXT,
                environment_vars TEXT,
                
                monthly_cost_usd REAL,
                total_cost_usd REAL DEFAULT 0,
                
                created_at TEXT,
                updated_at TEXT,
                deployed_at TEXT,
                deleted_at TEXT
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS deployment_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT,
                date TEXT NOT NULL,
                requests_count INTEGER DEFAULT 0,
                uptime_hours REAL DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                created_at TEXT,
                UNIQUE(deployment_id, date)
            )
        """)
        
        self.conn.commit()
    
    def create_deployment(self, deployment_id: str, user_id: str,
                         service_name: str, prompt: str,
                         provider: DeploymentProvider) -> Dict:
        """
        STEP 1: Track deployment creation
        Call this BEFORE calling Railway API
        """
        
        deployment = {
            'deployment_id': deployment_id,
            'user_id': user_id,
            'service_name': service_name,
            'prompt': prompt,
            'provider': provider.value,
            'status': DeploymentStatus.CREATING.value,
            'monthly_cost_usd': 5.0 if provider == DeploymentProvider.RAILWAY else 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        if self.use_postgres:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO deployments
                    (deployment_id, user_id, service_name, prompt, provider,
                     status, monthly_cost_usd, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """, (
                    deployment_id, user_id, service_name, prompt, provider.value,
                    DeploymentStatus.CREATING.value, 5.0
                ))
                self.conn.commit()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO deployments
                (deployment_id, user_id, service_name, prompt, provider,
                 status, monthly_cost_usd, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                deployment_id, user_id, service_name, prompt, provider.value,
                DeploymentStatus.CREATING.value, 5.0,
                datetime.now().isoformat(), datetime.now().isoformat()
            ))
            self.conn.commit()
        
        return deployment
    
    def update_external_ids(self, deployment_id: str,
                           project_id: str, service_id: str,
                           deployment_ext_id: Optional[str] = None):
        """
        STEP 2: Save Railway IDs after creation
        CRITICAL: Without this, we can't delete later!
        """
        
        if self.use_postgres:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE deployments
                    SET external_project_id = %s,
                        external_service_id = %s,
                        external_deployment_id = %s,
                        updated_at = NOW()
                    WHERE deployment_id = %s
                """, (project_id, service_id, deployment_ext_id, deployment_id))
                self.conn.commit()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE deployments
                SET external_project_id = ?,
                    external_service_id = ?,
                    external_deployment_id = ?,
                    updated_at = ?
                WHERE deployment_id = ?
            """, (project_id, service_id, deployment_ext_id,
                  datetime.now().isoformat(), deployment_id))
            self.conn.commit()
    
    def mark_deployed(self, deployment_id: str, service_url: str,
                     github_repo_url: str, endpoints: List[str]):
        """
        STEP 3: Mark as successfully deployed
        """
        
        endpoints_json = json.dumps(endpoints)
        
        if self.use_postgres:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE deployments
                    SET status = %s,
                        service_url = %s,
                        github_repo_url = %s,
                        endpoints = %s,
                        deployed_at = NOW(),
                        updated_at = NOW()
                    WHERE deployment_id = %s
                """, (DeploymentStatus.ACTIVE.value, service_url,
                      github_repo_url, Json(endpoints), deployment_id))
                self.conn.commit()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE deployments
                SET status = ?,
                    service_url = ?,
                    github_repo_url = ?,
                    endpoints = ?,
                    deployed_at = ?,
                    updated_at = ?
                WHERE deployment_id = ?
            """, (DeploymentStatus.ACTIVE.value, service_url,
                  github_repo_url, endpoints_json,
                  datetime.now().isoformat(),
                  datetime.now().isoformat(), deployment_id))
            self.conn.commit()
    
    def mark_failed(self, deployment_id: str, error: str):
        """Mark deployment as failed"""
        
        if self.use_postgres:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE deployments
                    SET status = %s, error_message = %s, updated_at = NOW()
                    WHERE deployment_id = %s
                """, (DeploymentStatus.FAILED.value, error, deployment_id))
                self.conn.commit()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE deployments
                SET status = ?, error_message = ?, updated_at = ?
                WHERE deployment_id = ?
            """, (DeploymentStatus.FAILED.value, error,
                  datetime.now().isoformat(), deployment_id))
            self.conn.commit()
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict]:
        """Get deployment by ID"""
        
        if self.use_postgres:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM deployments WHERE deployment_id = %s
                """, (deployment_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        else:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT * FROM deployments WHERE deployment_id = ?
            """, (deployment_id,))
            row = cur.fetchone()
            if not row:
                return None
            
            deployment = dict(row)
            # Parse JSON fields
            if deployment.get('endpoints'):
                deployment['endpoints'] = json.loads(deployment['endpoints'])
            if deployment.get('environment_vars'):
                deployment['environment_vars'] = json.loads(deployment['environment_vars'])
            return deployment
    
    def get_user_deployments(self, user_id: str,
                            status: Optional[DeploymentStatus] = None) -> List[Dict]:
        """Get all deployments for a user"""
        
        if self.use_postgres:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                if status:
                    cur.execute("""
                        SELECT * FROM deployments
                        WHERE user_id = %s AND status = %s
                        ORDER BY created_at DESC
                    """, (user_id, status.value))
                else:
                    cur.execute("""
                        SELECT * FROM deployments
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                    """, (user_id,))
                return [dict(row) for row in cur.fetchall()]
        else:
            cur = self.conn.cursor()
            if status:
                cur.execute("""
                    SELECT * FROM deployments
                    WHERE user_id = ? AND status = ?
                    ORDER BY created_at DESC
                """, (user_id, status.value))
            else:
                cur.execute("""
                    SELECT * FROM deployments
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
            
            rows = cur.fetchall()
            deployments = []
            for row in rows:
                d = dict(row)
                if d.get('endpoints'):
                    d['endpoints'] = json.loads(d['endpoints'])
                if d.get('environment_vars'):
                    d['environment_vars'] = json.loads(d['environment_vars'])
                deployments.append(d)
            return deployments
    
    def mark_deleting(self, deployment_id: str):
        """Mark deployment as being deleted"""
        
        if self.use_postgres:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE deployments
                    SET status = %s, updated_at = NOW()
                    WHERE deployment_id = %s
                """, (DeploymentStatus.DELETING.value, deployment_id))
                self.conn.commit()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE deployments
                SET status = ?, updated_at = ?
                WHERE deployment_id = ?
            """, (DeploymentStatus.DELETING.value,
                  datetime.now().isoformat(), deployment_id))
            self.conn.commit()
    
    def mark_deleted(self, deployment_id: str):
        """
        CRITICAL: Mark as deleted after Railway confirms deletion
        """
        
        if self.use_postgres:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE deployments
                    SET status = %s, deleted_at = NOW(), updated_at = NOW()
                    WHERE deployment_id = %s
                """, (DeploymentStatus.DELETED.value, deployment_id))
                self.conn.commit()
        else:
            cur = self.conn.cursor()
            now = datetime.now().isoformat()
            cur.execute("""
                UPDATE deployments
                SET status = ?, deleted_at = ?, updated_at = ?
                WHERE deployment_id = ?
            """, (DeploymentStatus.DELETED.value, now, now, deployment_id))
            self.conn.commit()
    
    def find_orphans(self) -> List[Dict]:
        """
        Find deployments that are ACTIVE in database but don't exist in Railway
        These are costing you money!
        """
        
        if self.use_postgres:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM deployments
                    WHERE status = %s
                    AND external_service_id IS NOT NULL
                    ORDER BY created_at DESC
                """, (DeploymentStatus.ACTIVE.value,))
                return [dict(row) for row in cur.fetchall()]
        else:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT * FROM deployments
                WHERE status = ?
                AND external_service_id IS NOT NULL
                ORDER BY created_at DESC
            """, (DeploymentStatus.ACTIVE.value,))
            return [dict(row) for row in cur.fetchall()]
    
    def calculate_costs(self, user_id: str) -> Dict:
        """Calculate user's deployment costs"""
        
        if self.use_postgres:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_deployments,
                        COUNT(*) FILTER (WHERE status = %s) as active_deployments,
                        SUM(monthly_cost_usd) FILTER (WHERE status = %s) as monthly_cost,
                        SUM(total_cost_usd) as total_spent
                    FROM deployments
                    WHERE user_id = %s
                """, (DeploymentStatus.ACTIVE.value,
                      DeploymentStatus.ACTIVE.value, user_id))
                return dict(cur.fetchone())
        else:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) as total_deployments,
                    SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as active_deployments,
                    SUM(CASE WHEN status = ? THEN monthly_cost_usd ELSE 0 END) as monthly_cost,
                    SUM(total_cost_usd) as total_spent
                FROM deployments
                WHERE user_id = ?
            """, (DeploymentStatus.ACTIVE.value,
                  DeploymentStatus.ACTIVE.value, user_id))
            row = cur.fetchone()
            return {
                'total_deployments': row[0] or 0,
                'active_deployments': row[1] or 0,
                'monthly_cost': row[2] or 0,
                'total_spent': row[3] or 0
            }


class LifecycleManager:
    """
    Complete lifecycle management with Railway sync
    
    Prevents orphaned services that cost money
    """
    
    def __init__(self, railway_token: str, tracker: DeploymentTracker):
        self.railway_token = railway_token
        self.tracker = tracker
        self.api_url = "https://backboard.railway.app/graphql/v2"
    
    def _railway_request(self, query: str, variables: Dict) -> Dict:
        """Make Railway API request"""
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.railway_token}",
                "Content-Type": "application/json"
            },
            json={"query": query, "variables": variables}
        )
        response.raise_for_status()
        return response.json()
    
    def delete_deployment(self, deployment_id: str) -> Dict:
        """
        COMPLETE DELETION - Database + Railway
        
        This is what prevents orphaned services!
        """
        
        # Step 1: Get deployment from database
        deployment = self.tracker.get_deployment(deployment_id)
        
        if not deployment:
            return {
                "success": False,
                "error": "Deployment not found"
            }
        
        if deployment['status'] == DeploymentStatus.DELETED.value:
            return {
                "success": True,
                "message": "Already deleted"
            }
        
        # Step 2: Mark as deleting
        self.tracker.mark_deleting(deployment_id)
        
        try:
            # Step 3: Delete from Railway
            if deployment.get('external_service_id'):
                mutation = """
                mutation DeleteService($serviceId: String!) {
                    serviceDelete(id: $serviceId)
                }
                """
                
                self._railway_request(mutation, {
                    "serviceId": deployment['external_service_id']
                })
            
            # Step 4: Delete project if it exists
            if deployment.get('external_project_id'):
                mutation = """
                mutation DeleteProject($projectId: String!) {
                    projectDelete(id: $projectId)
                }
                """
                
                self._railway_request(mutation, {
                    "projectId": deployment['external_project_id']
                })
            
            # Step 5: Mark as deleted in database
            self.tracker.mark_deleted(deployment_id)
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "deleted_from_railway": True,
                "deleted_from_database": True
            }
            
        except Exception as e:
            # Rollback status
            self.tracker.mark_failed(deployment_id, str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup_orphans(self, user_id: Optional[str] = None) -> Dict:
        """
        Find and delete orphaned services
        
        Run this periodically to prevent cost leaks
        """
        
        orphans = self.tracker.find_orphans()
        
        if user_id:
            orphans = [d for d in orphans if d['user_id'] == user_id]
        
        deleted = []
        failed = []
        
        for deployment in orphans:
            # Check if service actually exists in Railway
            try:
                query = """
                query GetService($serviceId: String!) {
                    service(id: $serviceId) {
                        id
                        name
                    }
                }
                """
                
                result = self._railway_request(query, {
                    "serviceId": deployment['external_service_id']
                })
                
                # If service doesn't exist in Railway, mark as deleted
                if not result.get('data', {}).get('service'):
                    self.tracker.mark_deleted(deployment['deployment_id'])
                    deleted.append(deployment['deployment_id'])
                
            except Exception as e:
                failed.append({
                    "deployment_id": deployment['deployment_id'],
                    "error": str(e)
                })
        
        return {
            "total_checked": len(orphans),
            "deleted": deleted,
            "failed": failed
        }


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    tracker = DeploymentTracker()
    
    # When user deploys:
    deployment = tracker.create_deployment(
        deployment_id="deploy_123",
        user_id="user_456",
        service_name="My API",
        prompt="Build payment API",
        provider=DeploymentProvider.RAILWAY
    )
    
    # After Railway creates service:
    tracker.update_external_ids(
        deployment_id="deploy_123",
        project_id="proj_abc",
        service_id="srv_xyz"
    )
    
    # After deployment succeeds:
    tracker.mark_deployed(
        deployment_id="deploy_123",
        service_url="https://my-api.railway.app",
        github_repo_url="https://github.com/user/my-api",
        endpoints=["POST /payments/create"]
    )
    
    # When user deletes:
    lifecycle = LifecycleManager(
        railway_token=os.getenv("RAILWAY_TOKEN"),
        tracker=tracker
    )
    
    result = lifecycle.delete_deployment("deploy_123")
    print(result)
    # Both Railway AND database deleted!
