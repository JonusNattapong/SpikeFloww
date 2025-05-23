#!/bin/bash

# SpikeFlow One-Click Deployment Script
# Handles the complete launch process

set -e  # Exit on any error

echo "🚀 SpikeFlow Launch Deployment Script"
echo "====================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# Check Git
if ! command -v git &> /dev/null; then
    print_error "Git not found. Please install Git"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run from SpikeFlow root directory"
    exit 1
fi

print_status "Prerequisites check passed"

# Install/upgrade build tools
echo "📦 Installing build dependencies..."
python -m pip install --upgrade pip build twine
print_status "Build tools installed"

# Install SpikeFlow in development mode
echo "🔧 Installing SpikeFlow in development mode..."
pip install -e .[dev,docs]
print_status "SpikeFlow installed"

# Run tests
echo "🧪 Running test suite..."
if python -m pytest tests/ -v; then
    print_status "All tests passed"
else
    print_error "Tests failed. Please fix before deploying"
    exit 1
fi

# Build documentation
echo "📚 Building documentation..."
cd docs
if make html; then
    print_status "Documentation built successfully"
    cd ..
else
    print_error "Documentation build failed"
    cd ..
    exit 1
fi

# Build package
echo "📦 Building distribution packages..."
python -m build
if [ $? -eq 0 ]; then
    print_status "Package built successfully"
else
    print_error "Package build failed"
    exit 1
fi

# Generate launch announcements
echo "📢 Generating launch announcements..."
python scripts/launch_announcement.py
print_status "Launch announcements generated"

# Ask for deployment confirmation
echo ""
echo "🚨 READY FOR LAUNCH! 🚨"
echo ""
echo "The following will be deployed:"
echo "- 📦 Python package to PyPI"
echo "- 📚 Documentation to GitHub Pages"
echo "- 🏷️  Git tag and release"
echo "- 📢 Launch announcements prepared"
echo ""

read -p "Continue with deployment? (y/N): " -n 1 -r
echo
if [[ ! $CORE =~ ^[Yy]$ ]]; then
    print_warning "Deployment cancelled by user"
    exit 0
fi

# Create git tag and push
echo "🏷️ Creating git tag..."
CURRENT_VERSION=$(python -c "import spikeflow; print(spikeflow.__version__)")
git add .
git commit -m "Prepare for v${CURRENT_VERSION} release" || true
git tag -a "v${CURRENT_VERSION}" -m "Release version ${CURRENT_VERSION}"
git push origin main
git push origin "v${CURRENT_VERSION}"
print_status "Git tag v${CURRENT_VERSION} created and pushed"

# Deploy to PyPI
echo "🚀 Deploying to PyPI..."
if python -m twine upload dist/*; then
    print_status "Successfully deployed to PyPI"
else
    print_error "PyPI deployment failed"
    exit 1
fi

# Success message
echo ""
echo "🎉 LAUNCH SUCCESSFUL! 🎉"
echo "======================="
echo ""
echo "SpikeFlow v${CURRENT_VERSION} has been successfully launched!"
echo ""
echo "📦 PyPI: https://pypi.org/project/spikeflow/"
echo "🐙 GitHub: https://github.com/JonusNattapong/SpikeFlow"
echo "📚 Docs: https://spikeflow.readthedocs.io"
echo ""
echo "Next steps:"
echo "1. 📢 Post launch announcements (see launch_announcements/ directory)"
echo "2. 🎬 Create GitHub release with release notes"
echo "3. 📊 Monitor PyPI downloads and GitHub stars"
echo "4. 🤝 Engage with the community"
echo ""
echo "Thank you for using SpikeFlow! 🧠⚡"
