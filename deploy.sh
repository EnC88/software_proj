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

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Function to build and start the application
start_app() {
    print_status "Building and starting the application..."
    docker-compose up --build -d
    print_success "Application started successfully!"
    print_status "Access the web interface at: http://localhost:7860"
    print_status "Access the Flask API at: http://localhost:5000"
}

# Function to stop the application
stop_app() {
    print_status "Stopping the application..."
    docker-compose down
    print_success "Application stopped successfully!"
}

# Function to view logs
view_logs() {
    print_status "Showing application logs..."
    docker-compose logs -f app
}

# Function to restart the application
restart_app() {
    print_status "Restarting the application..."
    docker-compose restart
    print_success "Application restarted successfully!"
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up Docker resources..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to show status
show_status() {
    print_status "Application status:"
    docker-compose ps
    echo ""
    print_status "Container logs (last 10 lines):"
    docker-compose logs --tail=10 app
}

# Function to show help
show_help() {
    echo "Software Compatibility Project Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Build and start the application"
    echo "  stop      Stop the application"
    echo "  restart   Restart the application"
    echo "  logs      View application logs"
    echo "  status    Show application status"
    echo "  cleanup   Remove all containers, images, and volumes"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start the application"
    echo "  $0 logs     # View logs"
    echo "  $0 stop     # Stop the application"
}

# Main script logic
case "${1:-help}" in
    start)
        check_docker
        start_app
        ;;
    stop)
        check_docker
        stop_app
        ;;
    restart)
        check_docker
        restart_app
        ;;
    logs)
        check_docker
        view_logs
        ;;
    status)
        check_docker
        show_status
        ;;
    cleanup)
        check_docker
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 