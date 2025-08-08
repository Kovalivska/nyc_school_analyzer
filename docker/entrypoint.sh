#!/bin/bash
# Entrypoint script for NYC School Analyzer Docker container
# Provides flexible execution modes and proper error handling

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${NYC_SCHOOLS_LOG_LEVEL}" == "DEBUG" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Function to display banner
show_banner() {
    cat << 'EOF'
    ███╗   ██╗██╗   ██╗ ██████╗    ███████╗ ██████╗██╗  ██╗ ██████╗  ██████╗ ██╗     
    ████╗  ██║╚██╗ ██╔╝██╔════╝    ██╔════╝██╔════╝██║  ██║██╔═══██╗██╔═══██╗██║     
    ██╔██╗ ██║ ╚████╔╝ ██║         ███████╗██║     ███████║██║   ██║██║   ██║██║     
    ██║╚██╗██║  ╚██╔╝  ██║         ╚════██║██║     ██╔══██║██║   ██║██║   ██║██║     
    ██║ ╚████║   ██║   ╚██████╗    ███████║╚██████╗██║  ██║╚██████╔╝╚██████╔╝███████╗
    ╚═╝  ╚═══╝   ╚═╝    ╚═════╝    ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝
                                                                                       
     █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗██████╗                  
    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝██╔══██╗                 
    ███████║██╔██╗ ██║███████║██║   ╚████╔╝   ███╔╝ █████╗  ██████╔╝                 
    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  ██╔══██╗                 
    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████╗███████╗██║  ██║                 
    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝                 
    
    Production-ready NYC High School Directory Analysis Tool
    Docker Container v1.0.0
EOF
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python installation
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check NYC School Analyzer installation
    if ! python -c "import nyc_school_analyzer" &> /dev/null; then
        log_error "NYC School Analyzer package is not properly installed"
        exit 1
    fi
    
    # Check required directories
    for dir in "/app/data" "/app/outputs" "/app/logs"; do
        if [[ ! -d "$dir" ]]; then
            log_warn "Directory $dir does not exist, creating..."
            mkdir -p "$dir"
        fi
    done
    
    log_info "Prerequisites check completed successfully"
}

# Function to set up environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Set default environment variables if not provided
    export NYC_SCHOOLS_LOG_LEVEL=${NYC_SCHOOLS_LOG_LEVEL:-INFO}
    export NYC_SCHOOLS_OUTPUT_PATH=${NYC_SCHOOLS_OUTPUT_PATH:-/app/outputs}
    export MPLBACKEND=${MPLBACKEND:-Agg}
    export PYTHONPATH=${PYTHONPATH:-/app/src}
    
    # Create logs directory if it doesn't exist
    mkdir -p /app/logs
    
    # Set permissions
    chmod 755 /app/outputs /app/logs
    
    log_debug "Environment variables:"
    log_debug "  NYC_SCHOOLS_LOG_LEVEL=$NYC_SCHOOLS_LOG_LEVEL"
    log_debug "  NYC_SCHOOLS_OUTPUT_PATH=$NYC_SCHOOLS_OUTPUT_PATH"
    log_debug "  MPLBACKEND=$MPLBACKEND"
    log_debug "  PYTHONPATH=$PYTHONPATH"
    
    log_info "Environment setup completed"
}

# Function to run analysis
run_analysis() {
    local data_file="$1"
    shift
    local additional_args="$@"
    
    log_info "Starting analysis of: $data_file"
    log_debug "Additional arguments: $additional_args"
    
    # Check if data file exists
    if [[ ! -f "$data_file" ]]; then
        log_error "Data file not found: $data_file"
        log_info "Available files in /app/data:"
        ls -la /app/data/
        exit 1
    fi
    
    # Run the analysis
    log_info "Executing: nyc-schools analyze $data_file $additional_args"
    python -m nyc_school_analyzer.cli.main analyze "$data_file" $additional_args
    
    if [[ $? -eq 0 ]]; then
        log_info "Analysis completed successfully"
        log_info "Results available in: $NYC_SCHOOLS_OUTPUT_PATH"
    else
        log_error "Analysis failed"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
NYC School Analyzer Docker Container

Usage:
  docker run [OPTIONS] nyc-school-analyzer [COMMAND] [ARGS...]

Commands:
  analyze <file>     Run comprehensive school data analysis
  visualize <file>   Generate visualizations from processed data
  validate <file>    Validate school data file
  export <file>      Export processed data in various formats
  config             Show current configuration
  version            Show version information
  help, --help       Show this help message

Analysis Examples:
  # Basic analysis
  docker run -v /path/to/data:/app/data -v /path/to/outputs:/app/outputs \\
    nyc-school-analyzer analyze /app/data/high-school-directory.csv

  # Analysis with specific borough and grade
  docker run -v /path/to/data:/app/data -v /path/to/outputs:/app/outputs \\
    nyc-school-analyzer analyze /app/data/schools.csv --borough BROOKLYN --grade 9

  # Generate visualizations only
  docker run -v /path/to/data:/app/data -v /path/to/outputs:/app/outputs \\
    nyc-school-analyzer visualize /app/data/processed_data.csv

Environment Variables:
  NYC_SCHOOLS_LOG_LEVEL     Set logging level (DEBUG, INFO, WARNING, ERROR)
  NYC_SCHOOLS_OUTPUT_PATH   Set output directory path
  NYC_SCHOOLS_TARGET_BOROUGH Set target borough for analysis
  NYC_SCHOOLS_GRADE         Set grade of interest

Volume Mounts:
  /app/data       Mount directory containing input data files
  /app/outputs    Mount directory for output files (charts, reports, exports)
  /app/logs       Mount directory for log files

For more information, visit: https://github.com/nyc-schools/nyc-school-analyzer
EOF
}

# Main execution logic
main() {
    # Show banner unless explicitly disabled
    if [[ "${DISABLE_BANNER}" != "true" ]]; then
        show_banner
        echo
    fi
    
    # Handle special commands
    case "${1:-}" in
        "help"|"--help"|"-h"|"")
            show_usage
            exit 0
            ;;
        "version"|"--version"|"-v")
            python -c "from nyc_school_analyzer import __version__; print(f'NYC School Analyzer v{__version__}')"
            exit 0
            ;;
        "config")
            check_prerequisites
            setup_environment
            python -m nyc_school_analyzer.cli.main config
            exit 0
            ;;
    esac
    
    # Check prerequisites and setup environment
    check_prerequisites
    setup_environment
    
    # Handle analysis commands
    case "${1:-}" in
        "analyze")
            if [[ $# -lt 2 ]]; then
                log_error "Missing data file argument for analyze command"
                show_usage
                exit 1
            fi
            run_analysis "${@:2}"
            ;;
        "visualize"|"validate"|"export")
            log_info "Executing command: nyc-schools $@"
            python -m nyc_school_analyzer.cli.main "$@"
            ;;
        *)
            # Check if first argument is a file path
            if [[ -f "${1:-}" ]]; then
                log_info "Detected data file argument, running analysis..."
                run_analysis "$@"
            else
                log_error "Unknown command or file not found: ${1:-}"
                show_usage
                exit 1
            fi
            ;;
    esac
}

# Trap signals for graceful shutdown
trap 'log_warn "Received termination signal, shutting down..."; exit 0' TERM INT

# Execute main function with all arguments
main "$@"