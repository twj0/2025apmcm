"""
Master Script to Run All Enhanced Models (Q2-Q5) and Generate Visualizations.

This script orchestrates the execution of all models with their ML enhancements
and generates comprehensive visualizations of the results.

Usage:
    python run_all_models.py --questions 2 3 4 5 --visualize
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import RESULTS_DIR, FIGURES_DIR, ensure_directories
from utils.data_exporter import ModelResultsManager
from visualization.viz_template import create_all_visualizations

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS_DIR / 'run_all_models.log')
        ]
    )


def run_q2_model(use_transformer: bool = True) -> bool:
    """Run Q2 Auto Trade Model.
    
    Args:
        use_transformer: Whether to use Transformer ML enhancement
        
    Returns:
        True if successful
    """
    try:
        logger.info("="*80)
        logger.info("RUNNING Q2: Auto Trade Analysis")
        logger.info("="*80)
        
        from models.q2_autos import run_q2_analysis
        run_q2_analysis(use_transformer=use_transformer)
        
        logger.info("Q2 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Q2 failed: {e}", exc_info=True)
        return False


def run_q3_model() -> bool:
    """Run Q3 Semiconductor Model.
    
    Returns:
        True if successful
    """
    try:
        logger.info("="*80)
        logger.info("RUNNING Q3: Semiconductor Analysis")
        logger.info("="*80)
        
        from models.q3_semiconductors import run_q3_analysis
        run_q3_analysis()
        
        logger.info("Q3 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Q3 failed: {e}", exc_info=True)
        return False


def run_q4_model(use_ml: bool = True) -> bool:
    """Run Q4 Tariff Revenue Model.
    
    Args:
        use_ml: Whether to use ML enhancements
        
    Returns:
        True if successful
    """
    try:
        logger.info("="*80)
        logger.info("RUNNING Q4: Tariff Revenue Analysis")
        logger.info("="*80)
        
        from models.q4_tariff_revenue import run_q4_analysis
        run_q4_analysis(use_ml=use_ml)
        
        logger.info("Q4 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Q4 failed: {e}", exc_info=True)
        return False


def run_q5_model() -> bool:
    """Run Q5 Macro/Financial Model.
    
    Returns:
        True if successful
    """
    try:
        logger.info("="*80)
        logger.info("RUNNING Q5: Macro/Financial Analysis")
        logger.info("="*80)
        
        from models.q5_macro_finance import run_q5_analysis
        run_q5_analysis()
        
        logger.info("Q5 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Q5 failed: {e}", exc_info=True)
        return False


def generate_visualizations() -> bool:
    """Generate all visualizations.
    
    Returns:
        True if successful
    """
    try:
        logger.info("="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80)
        
        all_figures = create_all_visualizations(RESULTS_DIR, FIGURES_DIR)
        
        for question, figures in all_figures.items():
            logger.info(f"{question.upper()}: Generated {len(figures)} figures")
            for fig_path in figures:
                logger.info(f"  - {fig_path.name}")
        
        logger.info("Visualizations completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        return False


def generate_summary_reports() -> None:
    """Generate summary reports for all questions."""
    logger.info("="*80)
    logger.info("GENERATING SUMMARY REPORTS")
    logger.info("="*80)
    
    for q_num in [2, 3, 4, 5]:
        try:
            manager = ModelResultsManager(q_num, RESULTS_DIR)
            
            # Register known methods
            q_dir = RESULTS_DIR / f'q{q_num}'
            if q_dir.exists():
                for method_dir in q_dir.iterdir():
                    if method_dir.is_dir():
                        manager.register_method(method_dir.name)
            
            # Generate summary
            summary_path = manager.generate_summary()
            logger.info(f"Q{q_num} summary: {summary_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate summary for Q{q_num}: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run all enhanced models and generate visualizations')
    parser.add_argument('--questions', nargs='+', type=int, choices=[2, 3, 4, 5],
                       default=[2, 3, 4, 5], help='Questions to run')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML enhancements')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    ensure_directories()
    
    start_time = datetime.now()
    logger.info("="*80)
    logger.info(f"STARTING MODEL EXECUTION PIPELINE")
    logger.info(f"Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Questions: {args.questions}")
    logger.info(f"ML Enhancements: {'DISABLED' if args.no_ml else 'ENABLED'}")
    logger.info("="*80)
    
    # Track results
    results = {}
    
    # Run models
    if 2 in args.questions:
        results['q2'] = run_q2_model(use_transformer=not args.no_ml)
    
    if 3 in args.questions:
        results['q3'] = run_q3_model()
    
    if 4 in args.questions:
        results['q4'] = run_q4_model(use_ml=not args.no_ml)
    
    if 5 in args.questions:
        results['q5'] = run_q5_model()
    
    # Generate visualizations
    if args.visualize:
        results['visualizations'] = generate_visualizations()
    
    # Generate summary reports
    generate_summary_reports()
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("="*80)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration: {duration}")
    logger.info("")
    logger.info("Results:")
    
    for task, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {task.upper()}: {status}")
    
    logger.info("")
    logger.info(f"Results Directory: {RESULTS_DIR}")
    logger.info(f"Figures Directory: {FIGURES_DIR}")
    logger.info("="*80)
    
    # Exit with appropriate code
    if all(results.values()):
        logger.info("All tasks completed successfully!")
        sys.exit(0)
    else:
        logger.warning("Some tasks failed. Check logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()
