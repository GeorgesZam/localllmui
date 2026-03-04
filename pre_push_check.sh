#!/bin/bash
# Pre-push verification script
# Run this before pushing to ensure everything is correct

echo "======================================"
echo "Pre-Push Verification Script"
echo "======================================"

ERRORS=0

# Check Python syntax
echo ""
echo "1. Checking Python syntax..."
python -m py_compile src/*.py pyi_*.py *.py 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Python syntax OK"
else
    echo "   ✗ Python syntax errors found"
    ERRORS=$((ERRORS+1))
fi

# Check required files exist
echo ""
echo "2. Checking required files..."
REQUIRED_FILES=(
    "LocalRAG.spec"
    "pyi_rth_customtkinter.py"
    "pyi_rth_llama_cpp.py"
    "src/windows_helper.py"
    "test_llama_cpp_windows.py"
    ".github/workflows/build.yml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file exists"
    else
        echo "   ✗ $file missing!"
        ERRORS=$((ERRORS+1))
    fi
done

# Check spec file contains required elements
echo ""
echo "3. Checking LocalRAG.spec configuration..."

if grep -q "windows_helper" LocalRAG.spec; then
    echo "   ✓ windows_helper in hiddenimports"
else
    echo "   ✗ windows_helper NOT in hiddenimports"
    ERRORS=$((ERRORS+1))
fi

if grep -q "pyi_rth_customtkinter.py" LocalRAG.spec; then
    echo "   ✓ customtkinter runtime hook included"
else
    echo "   ✗ customtkinter runtime hook NOT included"
    ERRORS=$((ERRORS+1))
fi

if grep -q "pyi_rth_llama_cpp.py" LocalRAG.spec; then
    echo "   ✓ llama_cpp runtime hook included"
else
    echo "   ✗ llama_cpp runtime hook NOT included"
    ERRORS=$((ERRORS+1))
fi

if grep -q "console=True" LocalRAG.spec; then
    echo "   ✓ Console mode enabled (for debugging)"
else
    echo "   ✗ Console mode NOT enabled"
    ERRORS=$((ERRORS+1))
fi

if grep -q "collect_llama_cpp_dlls" LocalRAG.spec; then
    echo "   ✓ llama_cpp DLL collection function present"
else
    echo "   ✗ llama_cpp DLL collection function NOT present"
    ERRORS=$((ERRORS+1))
fi

# Check source files for required imports
echo ""
echo "4. Checking source file imports..."

if grep -q "from windows_helper import force_window_focus" src/ui.py; then
    echo "   ✓ ui.py imports windows_helper"
else
    echo "   ✗ ui.py does NOT import windows_helper"
    ERRORS=$((ERRORS+1))
fi

# Check for timeout in embedding model loading
if grep -q "load_thread.join(timeout=" src/rag.py; then
    echo "   ✓ rag.py has timeout for embedding model"
else
    echo "   ✗ rag.py does NOT have timeout"
    ERRORS=$((ERRORS+1))
fi

# Check GitHub Actions workflow
echo ""
echo "5. Checking GitHub Actions workflow..."

if grep -q "LocalRAG.spec" .github/workflows/build.yml; then
    echo "   ✓ Workflow uses LocalRAG.spec"
else
    echo "   ✗ Workflow does NOT use LocalRAG.spec"
    ERRORS=$((ERRORS+1))
fi

if ! grep -q "\-\-onefile" .github/workflows/build.yml; then
    echo "   ✓ Workflow NOT using --onefile (good)"
else
    echo "   ✗ Workflow still uses --onefile (should use spec file)"
    ERRORS=$((ERRORS+1))
fi

# Final summary
echo ""
echo "======================================"
if [ $ERRORS -eq 0 ]; then
    echo "✓ ALL CHECKS PASSED"
    echo "======================================"
    echo ""
    echo "You are ready to push!"
    echo ""
    echo "Commands:"
    echo "  git add ."
    echo "  git commit -m 'fix: Windows .exe build issues'"
    echo "  git push"
    echo ""
    exit 0
else
    echo "✗ $ERRORS CHECK(S) FAILED"
    echo "======================================"
    echo ""
    echo "Please fix the errors above before pushing."
    echo ""
    exit 1
fi
