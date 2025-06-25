#!/bin/bash

# Software Compatibility Project Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f env.example ]; then
            cp env.example .env
            print_warning "Please edit .env file with your secure credentials before starting the application."
            print_warning "Generate secure passwords using:"
            print_warning "  POSTGRES_PASSWORD=\$(openssl rand -base64 32)"
            print_warning "  FLASK_SECRET_KEY=\$(openssl rand -base64 64)"
            return 1
        else
            print_error "env.example not found. Please create a .env file manually."
            return 1
        fi
    fi
    return 0
}

# Function to generate secure credentials
generate_credentials() {
    print_status "Generating secure credentials..."
    
    # Generate secure password and secret key
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    FLASK_SECRET_KEY=$(openssl rand -base64 64)
    
    # Create .env file with secure credentials
    cat > .env << EOF
# Database Configuration
POSTGRES_DB=compatibility_db
POSTGRES_USER=app_user
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
DATABASE_URL=postgresql://app_user:${POSTGRES_PASSWORD}@postgres:5432/compatibility_db

# Flask Configuration
FLASK_SECRET_KEY=${FLASK_SECRET_KEY}
FLASK_ENV=production

# Application Configuration
PYTHONPATH=/app
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# Redis Configuration (optional)
REDIS_URL=redis://redis:6379
EOF
    
    print_success "Secure credentials generated and saved to .env"
    print_warning "Keep your .env file secure and never commit it to version control!"
}

# Function to build and start the application
start_app() {
    print_status "Building and starting the application..."
    
    # Check if .env file exists
    if ! check_env_file; then
        print_error "Cannot start without proper environment configuration."
        print_status "Run './deploy.sh setup' to generate secure credentials."
        exit 1
    fi
    
    # Start the application
    docker-compose up --build -d
    
    if [ $? -eq 0 ]; then
        print_success "Application started successfully!"
        print_status "Access the web interface at: http://localhost:7860"
        print_status "Access the Flask API at: http://localhost:5000"
        print_status "Access the analytics dashboard at: http://localhost:8501"
        print_status "PostgreSQL database at: localhost:5432"
        print_status "Redis cache at: localhost:6379"
        
        # Wait a moment for services to start
        sleep 5
        
        # Check service health
        print_status "Checking service health..."
        if curl -f http://localhost:7860/ > /dev/null 2>&1; then
            print_success "Gradio interface is healthy"
        else
            print_warning "Gradio interface may still be starting up"
        fi
        
        if curl -f http://localhost:5000/health > /dev/null 2>&1; then
            print_success "Flask API is healthy"
        else
            print_warning "Flask API may still be starting up"
        fi
    else
        print_error "Failed to start application"
        exit 1
    fi
}

# Function to stop the application
stop_app() {
    print_status "Stopping the application..."
    docker-compose down
    print_success "Application stopped"
}

# Function to restart the application
restart_app() {
    print_status "Restarting the application..."
    docker-compose down
    docker-compose up --build -d
    print_success "Application restarted"
}

# Function to view logs
view_logs() {
    print_status "Showing application logs..."
    docker-compose logs -f app
}

# Function to check status
check_status() {
    print_status "Checking application status..."
    
    # Check if containers are running
    if docker-compose ps | grep -q "Up"; then
        print_success "Application is running"
        docker-compose ps
    else
        print_warning "Application is not running"
    fi
    
    # Check service health
    print_status "Checking service health..."
    
    if curl -f http://localhost:7860/ > /dev/null 2>&1; then
        print_success "Gradio interface: http://localhost:7860"
    else
        print_error "Gradio interface: Not responding"
    fi
    
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        print_success "Flask API: http://localhost:5000"
    else
        print_error "Flask API: Not responding"
    fi
    
    if curl -f http://localhost:8501/ > /dev/null 2>&1; then
        print_success "Analytics Dashboard: http://localhost:8501"
    else
        print_error "Analytics Dashboard: Not responding"
    fi
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, images, and data. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Build and start the application"
    echo "  stop      - Stop the application"
    echo "  restart   - Restart the application"
    echo "  logs      - View application logs"
    echo "  status    - Check application status"
    echo "  setup     - Generate secure credentials and create .env file"
    echo "  cleanup   - Remove all containers, images, and data"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start     # Start the application"
    echo "  $0 setup     # Generate secure credentials"
    echo "  $0 status    # Check if services are running"
}

# Main script logic
case "${1:-start}" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        restart_app
        ;;
    logs)
        view_logs
        ;;
    status)
        check_status
        ;;
    setup)
        generate_credentials
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 