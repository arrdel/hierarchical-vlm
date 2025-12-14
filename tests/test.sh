#!/bin/bash

###############################################################################
# HierarchicalVLM - Test Runner Script
# Runs all unit tests with detailed reporting
###############################################################################

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 HierarchicalVLM - Test Suite Runner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to project root
PROJECT_ROOT="/home/adelechinda/home/projects/HierarchicalVLM"
cd "$PROJECT_ROOT"

echo "📁 Project Root: $PROJECT_ROOT"
echo ""

# Check if conda environment exists
echo "🔍 Checking conda environment..."
if conda env list | grep -q "hierarchical_vlm"; then
    echo "✅ Found hierarchical_vlm environment"
else
    echo "❌ ERROR: hierarchical_vlm environment not found"
    exit 1
fi
echo ""

# Check if pytest is installed
echo "📦 Checking pytest..."
if ! conda run -n hierarchical_vlm pip list | grep -q pytest; then
    echo "⚠️  pytest not found. Installing..."
    conda run -n hierarchical_vlm pip install pytest pytest-cov -q
    echo "✅ pytest installed"
else
    echo "✅ pytest found"
fi

# Install optional pytest plugins quietly (ignore errors if already present)
echo "📦 Installing optional pytest plugins..."
conda run -n hierarchical_vlm pip install pytest-html pytest-xdist -q 2>/dev/null || true
echo ""

# Check if test files exist
echo "📋 Checking test files..."
if [ ! -d "tests" ]; then
    echo "❌ ERROR: tests directory not found"
    exit 1
fi

TEST_COUNT=$(find tests -name "test_*.py" -o -name "*_test.py" | wc -l)
echo "✅ Found $TEST_COUNT test files"
echo ""

# List test files
echo "📄 Test Files:"
find tests -name "test_*.py" -o -name "*_test.py" | while read file; do
    echo "   • $file"
done
echo ""

# Create output directory for test reports
mkdir -p test_reports
echo "📁 Test Report Directory: test_reports/"
echo ""

# Run pytest with verbose output and coverage
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 RUNNING TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run tests with pytest
echo "Running: pytest tests/ -v --tb=short --color=yes"
echo ""

conda run -n hierarchical_vlm pytest tests/ \
    -v \
    --tb=short \
    --color=yes \
    --junit-xml=test_reports/junit.xml \
    --html=test_reports/report.html \
    --self-contained-html \
    2>&1 | tee test_reports/test_output.log || {
    # If HTML plugin fails, run without it
    echo "⚠️  HTML plugin not available, running basic tests..."
    conda run -n hierarchical_vlm pytest tests/ \
        -v \
        --tb=short \
        --junit-xml=test_reports/junit.xml \
        2>&1 | tee test_reports/test_output.log
}

TEST_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Parse test results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    STATUS="PASSED"
    COLOR="\033[0;32m"  # Green
else
    echo "❌ SOME TESTS FAILED"
    STATUS="FAILED"
    COLOR="\033[0;31m"  # Red
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Extract test statistics from output
echo "📊 Test Statistics:"
PASSED=$(grep -c "PASSED" test_reports/test_output.log || true)
FAILED=$(grep -c "FAILED" test_reports/test_output.log || true)
SKIPPED=$(grep -c "SKIPPED" test_reports/test_output.log || true)

echo "   • Passed:  $PASSED"
echo "   • Failed:  $FAILED"
echo "   • Skipped: $SKIPPED"
echo ""

# Run with coverage report
echo "📈 Running coverage analysis..."
echo ""

conda run -n hierarchical_vlm pytest tests/ \
    --cov=hierarchicalvlm \
    --cov-report=html:test_reports/coverage_html \
    --cov-report=term-missing \
    --cov-report=xml:test_reports/coverage.xml \
    -q 2>&1 | tee -a test_reports/coverage.log || {
    # If coverage plugin fails, run basic coverage
    echo "⚠️  Coverage plugin not available, running basic coverage..."
    conda run -n hierarchical_vlm pytest tests/ \
        --cov=hierarchicalvlm \
        --cov-report=term \
        -q 2>&1 | tee -a test_reports/coverage.log
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄 Test Reports Generated:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "test_reports/junit.xml" ]; then
    echo "   ✅ test_reports/junit.xml - JUnit format (for CI/CD)"
fi

if [ -f "test_reports/report.html" ]; then
    echo "   ✅ test_reports/report.html - HTML test report"
    echo "      Open in browser: file://$PROJECT_ROOT/test_reports/report.html"
fi

if [ -d "test_reports/coverage_html" ]; then
    echo "   ✅ test_reports/coverage_html/ - Coverage report (HTML)"
    echo "      Open in browser: file://$PROJECT_ROOT/test_reports/coverage_html/index.html"
fi

if [ -f "test_reports/coverage.xml" ]; then
    echo "   ✅ test_reports/coverage.xml - Coverage (Cobertura format)"
fi

if [ -f "test_reports/test_output.log" ]; then
    echo "   ✅ test_reports/test_output.log - Full test output log"
fi

if [ -f "test_reports/coverage.log" ]; then
    echo "   ✅ test_reports/coverage.log - Coverage analysis log"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Status: $STATUS"
echo "Exit Code: $TEST_EXIT_CODE"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✨ All tests passed successfully!"
    echo ""
    echo "🎉 Your project is ready for production!"
    echo ""
    exit 0
else
    echo "⚠️  Some tests failed. Please review:"
    echo "   • test_reports/report.html - for detailed results"
    echo "   • test_reports/test_output.log - for error messages"
    echo ""
    exit 1
fi
