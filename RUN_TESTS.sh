#!/bin/bash

# Test Execution Guide for Efficient Attention Mechanisms
# ========================================================

echo "🧪 Running Efficient Attention Tests"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test directory
TEST_DIR="tests"
TEST_FILE="test_attention.py"

echo -e "${YELLOW}Prerequisites:${NC}"
echo "  - PyTorch installed"
echo "  - Pytest installed"
echo "  - All attention modules available"
echo ""

# Option 1: Run all tests
echo -e "${YELLOW}Option 1: Run all tests${NC}"
echo "  Command: pytest $TEST_DIR/$TEST_FILE -v"
echo ""

# Option 2: Run specific test class
echo -e "${YELLOW}Option 2: Run specific test class${NC}"
echo "  Example: pytest $TEST_DIR/$TEST_FILE::TestStridedAttention -v"
echo ""

# Option 3: Run specific test
echo -e "${YELLOW}Option 3: Run specific test${NC}"
echo "  Example: pytest $TEST_DIR/$TEST_FILE::TestStridedAttention::test_basic_forward -v"
echo ""

# Option 4: Run with coverage
echo -e "${YELLOW}Option 4: Run with coverage${NC}"
echo "  Command: pytest $TEST_DIR/$TEST_FILE --cov=hierarchicalvlm.attention --cov-report=html"
echo ""

# Option 5: Run performance tests
echo -e "${YELLOW}Option 5: Run efficiency benchmarks${NC}"
echo "  Command: pytest $TEST_DIR/$TEST_FILE::TestEfficiency -v -s"
echo ""

echo -e "${GREEN}Available Test Classes:${NC}"
echo "  1. TestStridedAttention"
echo "     - test_basic_forward"
echo "     - test_gradient_flow"
echo "     - test_different_strides"
echo "     - test_dropout"
echo ""
echo "  2. TestLocalGlobalAttention"
echo "     - test_basic_forward"
echo "     - test_gradient_flow"
echo "     - test_different_window_sizes"
echo "     - test_global_token_selection"
echo ""
echo "  3. TestCrossMemoryAttention"
echo "     - test_basic_forward"
echo "     - test_without_residual"
echo "     - test_fusion_ratio"
echo "     - test_gradient_flow"
echo ""
echo "  4. TestHierarchicalAttentionBlock"
echo "     - test_strided_attention_block"
echo "     - test_local_global_block"
echo "     - test_with_memory"
echo ""
echo "  5. TestPerformerAttention"
echo "     - test_basic_forward"
echo "     - test_long_sequence"
echo "     - test_different_kernels"
echo "     - test_gradient_flow"
echo ""
echo "  6. TestMambaLayer"
echo "     - test_basic_forward"
echo "     - test_long_sequence"
echo "     - test_different_state_sizes"
echo "     - test_gradient_flow"
echo ""
echo "  7. TestLinearAttentionBlock"
echo "     - test_performer_block"
echo "     - test_mamba_block"
echo ""
echo "  8. TestEfficiency"
echo "     - test_performer_efficiency"
echo "     - test_mamba_efficiency"
echo ""

echo -e "${GREEN}Quick Test Commands:${NC}"
echo ""
echo "# Test all attention mechanisms"
echo "pytest tests/test_attention.py -v"
echo ""
echo "# Test with detailed output"
echo "pytest tests/test_attention.py -vv -s"
echo ""
echo "# Run only gradient flow tests"
echo "pytest tests/test_attention.py -k \"gradient\" -v"
echo ""
echo "# Run only efficiency tests"
echo "pytest tests/test_attention.py::TestEfficiency -v -s"
echo ""
echo "# Generate coverage report"
echo "pytest tests/test_attention.py --cov=hierarchicalvlm.attention --cov-report=term-missing"
echo ""
echo "# Generate HTML coverage report"
echo "pytest tests/test_attention.py --cov=hierarchicalvlm.attention --cov-report=html"
echo "# Open: htmlcov/index.html"
echo ""

echo -e "${GREEN}Expected Test Results:${NC}"
echo "  Total Tests: 27"
echo "  Expected: All tests passing ✅"
echo ""

echo -e "${YELLOW}Running Examples:${NC}"
echo "  Command: python examples/attention_examples.py"
echo ""

echo -e "${YELLOW}Building Documentation:${NC}"
echo "  See: docs/ATTENTION.md"
echo "  See: ATTENTION_IMPLEMENTATION.md"
echo "  See: EFFICIENT_ATTENTION_COMPLETE.md"
echo ""

echo -e "${GREEN}Installation & Setup:${NC}"
echo ""
echo "# Install dependencies"
echo "pip install torch torchvision"
echo "pip install pytest pytest-cov"
echo ""
echo "# Run tests"
echo "cd /home/adelechinda/home/projects/HierarchicalVLM"
echo "pytest tests/test_attention.py -v"
echo ""

echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
