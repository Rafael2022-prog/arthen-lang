#!/usr/bin/env python3
"""
ARTHEN Language Test Runner
===========================

Advanced test runner for ARTHEN Native Language with comprehensive reporting,
performance analysis, and CI/CD integration.

Features:
- Comprehensive test execution
- Performance benchmarking
- Coverage analysis
- Security testing
- AI/ML model validation
- Multi-target blockchain testing
- Parallel test execution
- Detailed reporting

Version: 2.0.0
Author: ARTHEN Development Team
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ARTHEN-TestRunner')

class ARTHENTestRunner:
    """Advanced test runner for ARTHEN language"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.path.dirname(os.path.dirname(__file__)))
        self.tests_dir = self.project_root / "tests"
        self.reports_dir = self.tests_dir / "reports"
        self.coverage_dir = self.tests_dir / "htmlcov"
        
        # Ensure directories exist
        self.reports_dir.mkdir(exist_ok=True)
        self.coverage_dir.mkdir(exist_ok=True)
        
        # Test categories
        self.test_categories = {
            'unit': 'Unit tests for individual components',
            'integration': 'Integration tests for component interaction',
            'performance': 'Performance and benchmark tests',
            'ai': 'AI and ML functionality tests',
            'blockchain': 'Blockchain-specific tests',
            'security': 'Security and vulnerability tests'
        }
        
    def run_comprehensive_tests(self, 
                              categories: List[str] = None,
                              parallel: bool = True,
                              coverage: bool = True,
                              performance: bool = False,
                              security: bool = False) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        
        logger.info("="*80)
        logger.info("ARTHEN LANGUAGE COMPREHENSIVE TEST EXECUTION")
        logger.info("="*80)
        
        start_time = time.time()
        results = {}
        
        try:
            # 1. Setup test environment
            logger.info("Setting up test environment...")
            self._setup_test_environment()
            
            # 2. Run unit tests
            if not categories or 'unit' in categories:
                logger.info("Running unit tests...")
                results['unit_tests'] = self._run_unit_tests(parallel, coverage)
            
            # 3. Run integration tests
            if not categories or 'integration' in categories:
                logger.info("Running integration tests...")
                results['integration_tests'] = self._run_integration_tests()
            
            # 4. Run AI/ML tests
            if not categories or 'ai' in categories:
                logger.info("Running AI/ML tests...")
                results['ai_tests'] = self._run_ai_tests()
            
            # 5. Run blockchain tests
            if not categories or 'blockchain' in categories:
                logger.info("Running blockchain tests...")
                results['blockchain_tests'] = self._run_blockchain_tests()
            
            # 6. Run performance tests
            if performance or (categories and 'performance' in categories):
                logger.info("Running performance tests...")
                results['performance_tests'] = self._run_performance_tests()
            
            # 7. Run security tests
            if security or (categories and 'security' in categories):
                logger.info("Running security tests...")
                results['security_tests'] = self._run_security_tests()
            
            # 8. Generate comprehensive report
            logger.info("Generating comprehensive report...")
            report = self._generate_comprehensive_report(results, time.time() - start_time)
            
            # 9. Save results
            self._save_test_results(results, report)
            
            logger.info("="*80)
            logger.info("TEST EXECUTION COMPLETED")
            logger.info("="*80)
            
            return {
                'success': True,
                'results': results,
                'report': report,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _setup_test_environment(self):
        """Setup test environment"""
        # Install test requirements
        requirements_file = self.tests_dir / "test_requirements.txt"
        if requirements_file.exists():
            logger.info("Installing test requirements...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=False, capture_output=True)
        
        # Set environment variables
        os.environ['PYTHONPATH'] = str(self.project_root)
        os.environ['ARTHEN_TEST_MODE'] = 'true'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        # Set deterministic hashing for child processes
        os.environ['PYTHONHASHSEED'] = '0'
        # Limit threads to reduce nondeterminism
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        
    def _run_unit_tests(self, parallel: bool = True, coverage: bool = True) -> Dict[str, Any]:
        """Run unit tests"""
        cmd = [sys.executable, "-m", "pytest"]
        
        # Discover all tests in the tests directory (do not restrict to a single file)
        
        # Add markers for unit tests
        cmd.extend(["-m", "unit or not (integration or performance or slow)"])
        
        # Add parallel execution (only if pytest-xdist is available)
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(["-n", "auto"])
            except ImportError:
                logger.warning("pytest-xdist not available, running tests sequentially")
        
        # Add coverage (only if pytest-cov is available)
        if coverage:
            try:
                import pytest_cov
                cmd.extend([
                    "--cov=compiler",
                    "--cov=stdlib", 
                    "--cov=parser",
                    "--cov-report=html",
                    "--cov-report=term-missing"
                ])
            except ImportError:
                logger.warning("pytest-cov not available, skipping coverage")
        
        # Add reporting (only if pytest-html is available)
        try:
            import pytest_html
            cmd.extend([
                "--html=" + str(self.reports_dir / "unit_tests.html"),
                "--self-contained-html"
            ])
        except ImportError:
            logger.warning("pytest-html not available, skipping HTML report")
        
        # Add JSON reporting (only if pytest-json-report is available)
        try:
            import pytest_jsonreport
            cmd.extend([
                "--json-report",
                "--json-report-file=" + str(self.reports_dir / "unit_tests.json")
            ])
        except ImportError:
            logger.warning("pytest-json-report not available, skipping JSON report")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        cmd = [sys.executable, "-m", "pytest"]
        
        # Discover all tests in the tests directory (do not restrict to a single file)
        
        # Add markers for integration tests
        cmd.extend(["-m", "integration and not slow"])
        
        # Add reporting (only if pytest-html is available)
        try:
            import pytest_html
            cmd.extend([
                "--html=" + str(self.reports_dir / "integration_tests.html"),
                "--self-contained-html"
            ])
        except ImportError:
            logger.warning("pytest-html not available, skipping HTML report")
        
        # Add JSON reporting (only if pytest-json-report is available)
        try:
            import pytest_jsonreport
            cmd.extend([
                "--json-report",
                "--json-report-file=" + str(self.reports_dir / "integration_tests.json")
            ])
        except ImportError:
            logger.warning("pytest-json-report not available, skipping JSON report")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def _run_ai_tests(self) -> Dict[str, Any]:
        """Run AI/ML tests"""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_arthen_comprehensive.py"),
            "-m", "ai",
            "--html=" + str(self.reports_dir / "ai_tests.html"),
            "--json-report",
            "--json-report-file=" + str(self.reports_dir / "ai_tests.json")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def _run_blockchain_tests(self) -> Dict[str, Any]:
        """Run blockchain-specific tests"""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_arthen_comprehensive.py"),
            "-m", "blockchain",
            "--html=" + str(self.reports_dir / "blockchain_tests.html"),
            "--json-report",
            "--json-report-file=" + str(self.reports_dir / "blockchain_tests.json")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_arthen_comprehensive.py"),
            "-m", "performance",
            "--benchmark-only",
            "--benchmark-json=" + str(self.reports_dir / "benchmark.json"),
            "--html=" + str(self.reports_dir / "performance_tests.html")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def _run_security_tests(self) -> Dict[str, Any]:
        """Run security tests"""
        results = {}
        
        # Run security-marked tests
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_arthen_comprehensive.py"),
            "-m", "security",
            "--html=" + str(self.reports_dir / "security_tests.html")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        results['pytest'] = {
            'exit_code': result.returncode,
            'success': result.returncode == 0
        }
        
        # Run bandit security analysis
        try:
            bandit_cmd = [
                sys.executable, "-m", "bandit", 
                "-r", str(self.project_root / "compiler"),
                "-r", str(self.project_root / "stdlib"),
                "-f", "json",
                "-o", str(self.reports_dir / "bandit_report.json")
            ]
            
            bandit_result = subprocess.run(bandit_cmd, capture_output=True, text=True)
            results['bandit'] = {
                'exit_code': bandit_result.returncode,
                'success': bandit_result.returncode == 0
            }
        except Exception as e:
            results['bandit'] = {'error': str(e)}
        
        # Run safety check
        try:
            safety_cmd = [
                sys.executable, "-m", "safety", "check",
                "--json", "--output", str(self.reports_dir / "safety_report.json")
            ]
            
            safety_result = subprocess.run(safety_cmd, capture_output=True, text=True)
            results['safety'] = {
                'exit_code': safety_result.returncode,
                'success': safety_result.returncode == 0
            }
        except Exception as e:
            results['safety'] = {'error': str(e)}
        
        return results
    
    def _generate_comprehensive_report(self, results: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        # Analyze results
        for category, result in results.items():
            if isinstance(result, dict) and 'success' in result:
                if result['success']:
                    total_passed += 1
                else:
                    total_failed += 1
                total_tests += 1
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'execution_time': execution_time,
            'summary': {
                'total_test_categories': total_tests,
                'passed_categories': total_passed,
                'failed_categories': total_failed,
                'success_rate': success_rate
            },
            'categories': results,
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'project_root': str(self.project_root)
            },
            'arthen_info': {
                'language_version': '2.0.0',
                'compiler_version': '2.0.0',
                'stdlib_version': '2.0.0'
            }
        }
        
        return report
    
    def _save_test_results(self, results: Dict[str, Any], report: Dict[str, Any]):
        """Save test results and report"""
        
        # Save comprehensive report
        report_file = self.reports_dir / "comprehensive_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save raw results
        results_file = self.reports_dir / "raw_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        summary_file = self.reports_dir / "test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("ARTHEN LANGUAGE TEST SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Execution Time: {report['execution_time']:.2f} seconds\n")
            f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n")
            f.write(f"Categories Passed: {report['summary']['passed_categories']}\n")
            f.write(f"Categories Failed: {report['summary']['failed_categories']}\n")
            f.write(f"Total Categories: {report['summary']['total_test_categories']}\n\n")
            
            f.write("CATEGORY DETAILS:\n")
            f.write("-" * 20 + "\n")
            for category, result in results.items():
                status = "PASS" if result.get('success', False) else "FAIL"
                f.write(f"{category}: {status}\n")
        
        logger.info(f"Test results saved to: {self.reports_dir}")


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="ARTHEN Language Test Runner")
    
    parser.add_argument(
        '--categories', 
        nargs='+', 
        choices=['unit', 'integration', 'ai', 'blockchain', 'performance', 'security'],
        help='Test categories to run'
    )
    
    parser.add_argument(
        '--parallel', 
        action='store_true', 
        default=True,
        help='Run tests in parallel'
    )
    
    parser.add_argument(
        '--no-coverage', 
        action='store_true',
        help='Disable coverage reporting'
    )
    
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Include performance tests'
    )
    
    parser.add_argument(
        '--security', 
        action='store_true',
        help='Include security tests'
    )
    
    parser.add_argument(
        '--project-root',
        help='Project root directory'
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ARTHENTestRunner(args.project_root)
    
    # Run tests
    result = runner.run_comprehensive_tests(
        categories=args.categories,
        parallel=args.parallel,
        coverage=not args.no_coverage,
        performance=args.performance,
        security=args.security
    )
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()