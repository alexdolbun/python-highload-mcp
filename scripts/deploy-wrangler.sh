#!/bin/bash
# Wrangler Deployment Script for Python HighLoad MCP
# Optimized CI/CD deployment automation with performance monitoring

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/deploy_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] ENVIRONMENT

Deploy Python HighLoad MCP to Cloudflare Workers using Wrangler CLI

ENVIRONMENTS:
    dev         Deploy to development environment
    staging     Deploy to staging environment  
    prod        Deploy to production environment
    local       Start local development server

OPTIONS:
    -h, --help                Show this help message
    -v, --verbose             Enable verbose logging
    -t, --test                Run tests before deployment
    -b, --build               Force rebuild before deployment
    -c, --check               Run health checks after deployment
    -s, --secrets             Update secrets before deployment
    --dry-run                 Show what would be deployed without doing it
    --skip-tests              Skip running tests (not recommended)
    --performance-test        Run performance tests after deployment

EXAMPLES:
    $0 dev                    Deploy to development
    $0 staging -t -c          Deploy to staging with tests and health checks
    $0 prod --build --test    Deploy to production with full build and tests
    $0 local                  Start local development server
    
EOF
}

# Parse command line arguments
VERBOSE=false
RUN_TESTS=false
FORCE_BUILD=false
RUN_HEALTH_CHECKS=false
UPDATE_SECRETS=false
DRY_RUN=false
SKIP_TESTS=false
PERFORMANCE_TEST=false
ENVIRONMENT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -b|--build)
            FORCE_BUILD=true
            shift
            ;;
        -c|--check)
            RUN_HEALTH_CHECKS=true
            shift
            ;;
        -s|--secrets)
            UPDATE_SECRETS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --performance-test)
            PERFORMANCE_TEST=true
            shift
            ;;
        dev|staging|prod|local)
            ENVIRONMENT=$1
            shift
            ;;
        *)
            error "Unknown option: $1. Use -h for help."
            ;;
    esac
done

# Validate environment
if [[ -z "$ENVIRONMENT" ]]; then
    error "Environment is required. Use -h for help."
fi

if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod|local)$ ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be dev, staging, prod, or local."
fi

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Change to project root
cd "$PROJECT_ROOT"

log "Starting deployment to $ENVIRONMENT environment"
info "Project root: $PROJECT_ROOT"
info "Log file: $LOG_FILE"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if wrangler is installed
    if ! command -v wrangler &> /dev/null; then
        error "Wrangler CLI is not installed. Run: npm install -g wrangler"
    fi
    
    # Check if node is installed
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed"
    fi
    
    # Check if python is installed
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        error "Python is not installed"
    fi
    
    # Check if required files exist
    local required_files=("wrangler.toml" "requirements.txt")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required file not found: $file"
        fi
    done
    
    log "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Install/update Node dependencies
    if [[ -f "package.json" ]]; then
        npm install --silent
    else
        npm init -y > /dev/null
        npm install --save-dev @cloudflare/workers-types
    fi
    
    # Install Python dependencies if needed
    if [[ "$FORCE_BUILD" == "true" ]]; then
        log "Installing Python dependencies..."
        python3 -m pip install -r requirements.txt --quiet
    fi
    
    log "Environment setup complete"
}

# Run Python tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warning "Skipping tests as requested"
        return 0
    fi
    
    if [[ "$RUN_TESTS" == "true" || "$ENVIRONMENT" == "prod" ]]; then
        log "Running Python tests..."
        
        # Install test dependencies
        python3 -m pip install pytest pytest-cov pytest-xdist --quiet
        
        # Run tests with parallel execution
        if [[ -d "tests" ]]; then
            pytest tests/ -v -x --tb=short --cov=src --cov-fail-under=70 -n auto
        else
            warning "No tests directory found, skipping tests"
        fi
        
        log "Tests completed successfully"
    fi
}

# Build project
build_project() {
    if [[ "$FORCE_BUILD" == "true" ]]; then
        log "Building project..."
        
        # Create Worker JavaScript if it doesn't exist
        if [[ ! -f "src/worker.js" ]]; then
            mkdir -p src
            cat > src/worker.js << 'EOF'
/**
 * Cloudflare Worker for Python HighLoad MCP Server
 * Auto-generated by deployment script
 */
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'healthy',
        version: env.PYTHON_MCP_VERSION || '1.0.0',
        timestamp: new Date().toISOString(),
        environment: env.ENVIRONMENT || 'unknown'
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    return new Response(JSON.stringify({
      message: 'Python HighLoad MCP Server',
      status: 'running',
      environment: env.ENVIRONMENT || 'unknown'
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
EOF
        fi
        
        log "Project build complete"
    fi
}

# Update secrets
update_secrets() {
    if [[ "$UPDATE_SECRETS" == "true" ]]; then
        log "Updating secrets..."
        
        # Check if secrets are configured
        if [[ -n "${CLOUDFLARE_API_TOKEN:-}" ]]; then
            info "CLOUDFLARE_API_TOKEN is configured"
        else
            warning "CLOUDFLARE_API_TOKEN not found in environment"
        fi
        
        log "Secrets update complete"
    fi
}

# Deploy to environment
deploy() {
    local env_name="$ENVIRONMENT"
    
    if [[ "$env_name" == "local" ]]; then
        start_local_development
        return 0
    fi
    
    log "Deploying to $env_name environment..."
    
    # Map environment names
    case "$env_name" in
        "dev")
            env_name="development"
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would deploy to $env_name environment"
        wrangler deploy --env "$env_name" --dry-run
        return 0
    fi
    
    # Deploy using Wrangler
    local version=$(date +'%Y.%m.%d')-$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    wrangler deploy --env "$env_name" --var "PYTHON_MCP_VERSION:$version"
    
    log "Deployment to $env_name completed successfully"
    
    # Save deployment info
    cat > "logs/last_deployment_${env_name}.json" << EOF
{
  "environment": "$env_name",
  "version": "$version",
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "deployed_by": "$(whoami)",
  "log_file": "$LOG_FILE"
}
EOF
}

# Start local development server
start_local_development() {
    log "Starting local development server..."
    
    info "Development server will be available at: http://localhost:8787"
    info "Press Ctrl+C to stop the server"
    
    # Start Wrangler dev server
    wrangler dev --env development --port 8787 --ip 127.0.0.1
}

# Run health checks
run_health_checks() {
    if [[ "$RUN_HEALTH_CHECKS" == "true" && "$ENVIRONMENT" != "local" ]]; then
        log "Running health checks..."
        
        sleep 10  # Wait for deployment to propagate
        
        # Determine URL based on environment
        local base_url
        case "$ENVIRONMENT" in
            "dev")
                base_url="https://python-highload-mcp-dev.YOUR_SUBDOMAIN.workers.dev"
                ;;
            "staging")
                base_url="https://python-highload-mcp-staging.YOUR_SUBDOMAIN.workers.dev"
                ;;
            "prod")
                base_url="https://python-highload-mcp-prod.YOUR_SUBDOMAIN.workers.dev"
                ;;
        esac
        
        info "Testing health endpoint: $base_url/health"
        
        # Test health endpoint
        local response
        if response=$(curl -f -s "$base_url/health" 2>/dev/null); then
            log "Health check passed: $response"
        else
            error "Health check failed for $base_url/health"
        fi
        
        log "Health checks completed"
    fi
}

# Run performance tests
run_performance_tests() {
    if [[ "$PERFORMANCE_TEST" == "true" && "$ENVIRONMENT" != "local" ]]; then
        log "Running performance tests..."
        
        # Simple performance test with curl
        local base_url
        case "$ENVIRONMENT" in
            "dev")
                base_url="https://python-highload-mcp-dev.YOUR_SUBDOMAIN.workers.dev"
                ;;
            "staging")
                base_url="https://python-highload-mcp-staging.YOUR_SUBDOMAIN.workers.dev"
                ;;
            "prod")
                base_url="https://python-highload-mcp-prod.YOUR_SUBDOMAIN.workers.dev"
                ;;
        esac
        
        info "Running performance test against: $base_url"
        
        # Run 10 requests and measure response time
        for i in {1..10}; do
            curl -w "Response time: %{time_total}s\n" -o /dev/null -s "$base_url/health"
        done
        
        log "Performance tests completed"
    fi
}

# Main execution
main() {
    log "Python HighLoad MCP Deployment Script Starting"
    
    check_prerequisites
    setup_environment
    run_tests
    build_project
    update_secrets
    deploy
    run_health_checks
    run_performance_tests
    
    log "Deployment completed successfully!"
    
    # Show deployment summary
    echo
    echo -e "${GREEN}=== Deployment Summary ===${NC}"
    echo "Environment: $ENVIRONMENT"
    echo "Timestamp: $(date)"
    echo "Log file: $LOG_FILE"
    
    if [[ "$ENVIRONMENT" != "local" ]]; then
        case "$ENVIRONMENT" in
            "dev")
                echo "URL: https://python-highload-mcp-dev.YOUR_SUBDOMAIN.workers.dev"
                ;;
            "staging")
                echo "URL: https://python-highload-mcp-staging.YOUR_SUBDOMAIN.workers.dev"
                ;;
            "prod")
                echo "URL: https://python-highload-mcp-prod.YOUR_SUBDOMAIN.workers.dev"
                ;;
        esac
    fi
}

# Execute main function
main "$@"
