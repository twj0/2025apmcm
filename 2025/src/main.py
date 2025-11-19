"""
Main entry point for APMCM 2025 Problem C analysis.

This script runs all question analyses (Q1-Q5) in sequence.

Usage:
    uv run python 2025/src/main.py [--questions Q1 Q2 Q3]
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    run_q1_analysis,
    run_q2_analysis,
    run_q3_analysis,
    run_q4_analysis,
    run_q5_analysis,
)
from utils.config import ensure_directories, set_random_seed
from utils.mapping import save_mapping_tables


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('2025/results/logs/analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the analysis environment."""
    logger.info("Setting up analysis environment")
    
    # Ensure directories exist
    ensure_directories()
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create mapping tables
    try:
        save_mapping_tables()
        logger.info("Mapping tables created")
    except Exception as e:
        logger.warning(f"Could not create mapping tables: {e}")


def run_all_analyses():
    """Run all question analyses in sequence."""
    analyses = {
        'Q1': ('Soybean Trade Analysis', run_q1_analysis),
        'Q2': ('Auto Trade Analysis', run_q2_analysis),
        'Q3': ('Semiconductor Analysis', run_q3_analysis),
        'Q4': ('Tariff Revenue Analysis', run_q4_analysis),
        'Q5': ('Macro/Financial Impact Analysis', run_q5_analysis),
    }
    
    for q_name, (description, analysis_func) in analyses.items():
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Starting {q_name}: {description}")
        logger.info("=" * 70)
        
        try:
            analysis_func()
            logger.info(f"✓ {q_name} completed successfully")
        except Exception as e:
            logger.error(f"✗ {q_name} failed with error: {e}", exc_info=True)
        
        logger.info("")


def run_selected_analyses(questions: list):
    """Run selected question analyses.
    
    Args:
        questions: List of question names (e.g., ['Q1', 'Q3'])
    """
    analysis_map = {
        'Q1': run_q1_analysis,
        'Q2': run_q2_analysis,
        'Q3': run_q3_analysis,
        'Q4': run_q4_analysis,
        'Q5': run_q5_analysis,
    }
    
    for q in questions:
        q_upper = q.upper()
        if q_upper not in analysis_map:
            logger.warning(f"Unknown question: {q}. Skipping.")
            continue
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Running {q_upper}")
        logger.info("=" * 70)
        
        try:
            analysis_map[q_upper]()
            logger.info(f"✓ {q_upper} completed")
        except Exception as e:
            logger.error(f"✗ {q_upper} failed: {e}", exc_info=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='APMCM 2025 Problem C Analysis'
    )
    parser.add_argument(
        '--questions',
        nargs='+',
        choices=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'q1', 'q2', 'q3', 'q4', 'q5'],
        help='Specific questions to run (default: all)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("APMCM 2025 Problem C - U.S. Tariff Policy Analysis")
    logger.info("=" * 70)
    
    # Setup
    setup_environment()
    
    # Run analyses
    if args.questions:
        run_selected_analyses(args.questions)
    else:
        run_all_analyses()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Analysis pipeline completed")
    logger.info("Results saved to: 2025/results/")
    logger.info("Figures saved to: 2025/figures/")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
